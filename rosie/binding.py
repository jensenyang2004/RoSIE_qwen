"""
Text binding for Qwen-Image-2.1: mice's final binding (apitasmkernelnonlap + free_latent + free_context, every
block, every step) on the joint sequence of rosie/tokens.py.

Allowed (query -> key), then AND the model's own block-causal rule (q >= k, or both in the same image):

    global text       -> global text, every image token
    source token      -> global text, every image token, its instance's prompt
    instance prompt k -> instance prompt k, the source and target tokens of instance k
    target, bg        -> global text, every image token
    target, inst k    -> instance prompt k, every image token

Image-to-image attention is never restricted (free_latent + free_context). Under the causal rule the global text
and the source never see the instance prompts or the target anyway, so in practice the binding only restricts the
instance prompts' rows (no other prompt, no global text, only their instance in the source) and the target's rows
(an instance reads only its own prompt; the background reads only the global prompt).

The model computes the text and source once at step 0 and caches their K/V; the processor uses the full mask then,
and its target rows on the cached steps.
"""
from contextlib import contextmanager

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

from qwen21.transformer_qwenimage21 import _qwenimage21_prepare_qkv
from rosie.tokens import Layout


def causal_mask(L: Layout) -> torch.Tensor:
    """The model's own rule, (n, n) bool: (q >= k) or same image block."""
    q = torch.arange(L.n)
    ids = L.image_ids
    return (q[:, None] >= q[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0))


def allowed_mask(L: Layout, bind: bool = True) -> torch.Tensor:
    """(n, n) bool, True = may attend. bind=False gives the model's own rule (for checks)."""
    allowed = causal_mask(L)
    if not bind:
        return allowed
    n = L.n
    role = torch.full((n,), -1, dtype=torch.long)  # instance of the token, -1 = none
    is_local = torch.zeros(n, dtype=torch.bool)
    for k, sl in enumerate(L.local):
        role[sl] = k
        is_local[sl] = True
    src, tgt = torch.arange(L.src.start, L.src.stop), torch.arange(L.tgt.start, L.tgt.stop)
    for k, idx in enumerate(L.inst):
        role[src[idx]] = k
        role[tgt[idx]] = k
    is_img = L.image_ids >= 0
    is_src, is_tgt = L.image_ids == 0, L.image_ids == 1
    is_glob = ~is_img & ~is_local

    key_glob, key_img = is_glob[None, :], is_img[None, :]
    same = role[:, None] == role[None, :]
    key_own_local = is_local[None, :] & same  # the query's own instance prompt (role -1 never matches)
    rule = torch.zeros(n, n, dtype=torch.bool)
    rule |= is_glob[:, None] & (key_glob | key_img)
    rule |= is_src[:, None] & (key_glob | key_img | key_own_local)
    rule |= is_local[:, None] & same & (is_local[None, :] | key_img)
    rule |= (is_tgt & (role < 0))[:, None] & (key_glob | key_img)
    rule |= (is_tgt & (role >= 0))[:, None] & (key_img | key_own_local)
    return allowed & rule


class BindAttnProcessor:
    """QwenImage21AttnProcessor with one explicit (n, n) mask instead of the model's per-segment prefill.

    Step 0 (prefill, every token is a query): the full mask. Cached steps (only the target's queries, keys =
    [cached prefix, target]): its target rows. Without the KV cache every step is a prefill.
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self, allowed: torch.Tensor, n_tgt: int):
        self.allowed = allowed
        self.n, self.n_tgt = allowed.shape[0], n_tgt
        self._dev = {}

    def _mask(self, device, rows):
        key = (device, rows)
        if key not in self._dev:
            m = self.allowed if rows == "all" else self.allowed[-self.n_tgt:]
            self._dev[key] = m.to(device)[None, None]
        return self._dev[key]

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None,
                 kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        assert key_valid is None or bool(key_valid.all()), "padded prompts are not supported"
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        if seq_len_q == self.n and key.shape[1] == self.n:
            mask = self._mask(query.device, "all")
        elif seq_len_q == self.n_tgt and key.shape[1] == self.n:
            mask = self._mask(query.device, "tgt")
        else:
            raise ValueError(f"sequence ({seq_len_q} queries, {key.shape[1]} keys) does not match the layout "
                             f"(n={self.n}, target={self.n_tgt})")
        out = dispatch_attention_fn(query, key, value, attn_mask=mask, dropout_p=0.0,
                                    backend=self._attention_backend, parallel_config=self._parallel_config)
        out = out.flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


@contextmanager
def binding(transformer, allowed: torch.Tensor, n_tgt: int):
    """Install BindAttnProcessor on every block for one sample, then put the previous processors back."""
    previous = transformer.attn_processors
    transformer.set_attn_processor(BindAttnProcessor(allowed, n_tgt))
    try:
        yield
    finally:
        transformer.set_attn_processor(previous)

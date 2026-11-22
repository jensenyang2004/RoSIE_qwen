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

The model computes the text and source once at step 0 and caches their K/V; the processor applies the mask's rows
for every token then, and its target rows on the cached steps. It never puts the (n, n) mask on the GPU: rows that
read the same keys attend together to just those keys (attention_groups), so most of the work runs as unmasked
(flash) attention, like the model's own per-segment prefill.
"""
from contextlib import contextmanager

import numpy as np
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


def attention_groups(allowed: torch.Tensor, rows: torch.Tensor) -> list:
    """Split the query rows into groups that each attend to one key set, so attention runs per group on just its
    keys (flash attention, no (n, n) mask on the GPU), as the model's own prefill does per segment.

    Rows with identical key sets form one group, wherever they are (an image block, an instance's target tokens).
    Runs of consecutive rows whose sets all differ (causal text) are merged, with a small mask over their keys.
    Returns [(row positions into `rows`, key positions, bool mask [len rows, len keys] or None)].
    """
    A = allowed[rows].numpy()
    packed = np.packbits(A, axis=1)
    by_set = {}
    for i, b in enumerate(packed):
        by_set.setdefault(b.tobytes(), []).append(i)
    groups, run = [], []
    for members in sorted(by_set.values(), key=lambda g: g[0]):
        if len(members) == 1 and (not run or members[0] == run[-1] + 1):
            run.append(members[0])
            continue
        if run:
            groups.append(run)
            run = []
        if len(members) == 1:
            run = members
        else:
            groups.append(members)
    if run:
        groups.append(run)
    out = []
    for g in groups:
        sub = A[g]
        keys = sub.any(0).nonzero()[0]
        mask = sub[:, keys]
        out.append((torch.as_tensor(g), torch.as_tensor(keys), None if mask.all() else torch.as_tensor(mask)))
    return out


def _take(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x[:, idx], a view when idx is a contiguous range."""
    if len(idx) and int(idx[-1]) - int(idx[0]) + 1 == len(idx):
        return x[:, int(idx[0]):int(idx[-1]) + 1]
    return x[:, idx]


class BindAttnProcessor:
    """QwenImage21AttnProcessor with the binding mask, computed per group of query rows (attention_groups).

    Step 0 (prefill, every token is a query): groups over all rows. Cached steps (only the target's queries, keys =
    [cached prefix, target]): groups over the target rows. Without the KV cache every step is a prefill.
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self, allowed: torch.Tensor, n_tgt: int):
        self.n, self.n_tgt = allowed.shape[0], n_tgt
        self.groups = {"all": attention_groups(allowed, torch.arange(self.n)),
                       "tgt": attention_groups(allowed, torch.arange(self.n - n_tgt, self.n))}
        self._dev = {}

    def _groups(self, device, rows):
        if (device, rows) not in self._dev:
            self._dev[device, rows] = [(q.to(device), k.to(device), None if m is None else m.to(device)[None, None])
                                       for q, k, m in self.groups[rows]]
        return self._dev[device, rows]

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None,
                 kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        assert key_valid is None or bool(key_valid.all()), "padded prompts are not supported"
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        if seq_len_q == self.n and key.shape[1] == self.n:
            rows = "all"
        elif seq_len_q == self.n_tgt and key.shape[1] == self.n:
            rows = "tgt"
        else:
            raise ValueError(f"sequence ({seq_len_q} queries, {key.shape[1]} keys) does not match the layout "
                             f"(n={self.n}, target={self.n_tgt})")
        out = torch.empty_like(query)
        for q_idx, k_idx, mask in self._groups(query.device, rows):
            out[:, q_idx] = dispatch_attention_fn(
                _take(query, q_idx), _take(key, k_idx), _take(value, k_idx), attn_mask=mask, dropout_p=0.0,
                backend=self._attention_backend if mask is None else None, parallel_config=self._parallel_config)
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

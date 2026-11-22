"""
Checks for the token assignment and the text binding.

    python check_binding.py            CPU, no weights, a tiny random transformer on a synthetic layout:
        1. bind off: BindAttnProcessor == the model's own processor (step 0 and cached steps)
        2. bind on, cached steps == recomputing everything (the target rows of the mask are the right ones)
        3. the attention groups rebuild the mask exactly (step 0 rows and cached-step rows)
        4. bind on, 1 block: changing instance prompt 2 changes only instance 2's target tokens; changing the
           global prompt changes only the background's; every instance prompt row reads only its own prompt and
           its instance's source tokens
    python check_binding.py --tokens   the real Qwen processor (downloads ~12 MB, no weights) on bench samples:
        layout sizes, the target words decoded from their positions must equal the bench targets, and the
        attention groups of the real layout rebuild its mask exactly
"""
import argparse
import sys

import torch

from rosie.binding import allowed_mask, attention_groups, binding
from rosie.tokens import Segment, build_layout


def synthetic_layout():
    """Global: 3 text, 4 slots (4x4 source), 5 text; instance prompts of 4 and 3 tokens; instances = the top-left
    and bottom-right 2x2 of the 4x4 grid."""
    slot = torch.tensor([False] * 3 + [True] * 4 + [False] * 5)
    glob = Segment(torch.zeros(12, dtype=torch.long), slot, slice(0, 12))
    locs = [Segment(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.bool), slice(0, 4), [], [2, 3]),
            Segment(torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool), slice(0, 3), [], [2])]
    m0, m1 = torch.zeros(4, 4, dtype=torch.bool), torch.zeros(4, 4, dtype=torch.bool)
    m0[:2, :2] = True
    m1[2:, 2:] = True
    return build_layout(glob, locs, [m0, m1], 4, 4)


def tiny_model(num_layers):
    from qwen21.transformer_qwenimage21 import QwenImage21Transformer2DModel
    torch.manual_seed(0)
    m = QwenImage21Transformer2DModel(num_layers=num_layers, attention_head_dim=16, num_attention_heads=2,
                                      context_in_dim=32, in_channels=8, out_channels=8, axes_dims_rope=(4, 6, 6))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return m.eval()


def inputs(L, gen):
    n_vlm = len(L.img_mask)
    return dict(hidden_states=torch.randn(1, 2 * L.H * L.W, 8, generator=gen),
                encoder_hidden_states=torch.randn(1, n_vlm, 32, generator=gen),
                img_shapes=[[(1, L.H, L.W), (1, L.H, L.W)]],
                img_mask=torch.cat([L.img_mask, torch.ones(L.H * L.W // 4, dtype=torch.bool)])[None])


def run(model, x, steps, cache):
    from qwen21.transformer_qwenimage21 import QwenImage21KVCache
    kv = QwenImage21KVCache(len(model.transformer_blocks)) if cache else None
    outs = []
    for i, t in enumerate(steps):
        mode = ("extract" if i == 0 else "cached") if cache else None
        outs.append(model(**x, timestep=torch.tensor([t]), kv_cache=kv, kv_cache_mode=mode, return_dict=False)[0])
    return outs


def groups_rebuild(allowed, n_tgt: int) -> tuple:
    """(exact, [number of groups at step 0, on cached steps])."""
    n = allowed.shape[0]
    exact, counts = True, []
    for rows in (torch.arange(n), torch.arange(n - n_tgt, n)):
        groups = attention_groups(allowed, rows)
        dense = torch.zeros(len(rows), n, dtype=torch.bool)
        for q, k, m in groups:
            dense[q[:, None], k] = True if m is None else m
        exact &= torch.equal(dense, allowed[rows])
        counts.append(len(groups))
    return exact, counts


def check_tiny() -> bool:
    L = synthetic_layout()
    T = L.H * L.W
    x = inputs(L, torch.Generator().manual_seed(1))
    ok = True

    def report(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {name} {extra}")

    for bind in (False, True):
        exact, counts = groups_rebuild(allowed_mask(L, bind), T)
        report(f"attention groups rebuild the mask (bind {bind})", exact, f"groups {counts}")

    with torch.no_grad():
        model = tiny_model(2)
        for cache in (False, True):
            ref = run(model, x, [0.9, 0.5], cache)
            with binding(model, allowed_mask(L, bind=False), T):
                got = run(model, x, [0.9, 0.5], cache)
            d = max((a - b).abs().max().item() for a, b in zip(ref, got))
            report(f"bind off == model's processor (kv cache {cache})", d < 1e-5, f"max|diff|={d:.2g}")

        with binding(model, allowed_mask(L), T):
            full = run(model, x, [0.9, 0.5], cache=False)[1][:, -T:]
            cached = run(model, x, [0.9, 0.5], cache=True)[1]
        d = (full - cached).abs().max().item()
        report("bind on: cached step == recompute", d < 1e-5, f"max|diff|={d:.2g}")

        one = tiny_model(1)
        allowed = allowed_mask(L)
        inst0, inst1 = L.inst
        bg = torch.tensor([i for i in range(T) if i not in set(inst0.tolist()) | set(inst1.tolist())])
        n_glob_vlm = 12
        loc1 = slice(n_glob_vlm + 4, n_glob_vlm + 7)  # instance prompt 2, in encoder_hidden_states positions
        glob_text = torch.tensor([i for i in range(n_glob_vlm) if not L.img_mask[i]])

        def target_out(xx):
            with binding(one, allowed, T):
                return one(**xx, timestep=torch.tensor([0.9]), return_dict=False)[0][0, -T:]

        base = target_out(x)
        x2 = dict(x, encoder_hidden_states=x["encoder_hidden_states"].clone())
        x2["encoder_hidden_states"][0, loc1] += 3.0
        d = (target_out(x2) - base).abs().amax(-1)
        report("instance prompt 2 changed: instance 1 and background unchanged",
               d[inst0].max() == 0 and d[bg].max() == 0, f"(max {max(d[inst0].max(), d[bg].max()):.2g})")
        report("instance prompt 2 changed: instance 2 changed", d[inst1].min() > 0, f"(min {d[inst1].min():.2g})")
        x3 = dict(x, encoder_hidden_states=x["encoder_hidden_states"].clone())
        x3["encoder_hidden_states"][0, glob_text] += 3.0
        d = (target_out(x3) - base).abs().amax(-1)
        report("global prompt changed: instances unchanged, background changed",
               d[torch.cat([inst0, inst1])].max() == 0 and d[bg].min() > 0)

        src = torch.arange(L.src.start, L.src.stop)
        for k, sl in enumerate(L.local):
            own = set(range(sl.start, sl.stop)) | set(src[L.inst[k]].tolist())
            keys = set(allowed[sl].any(0).nonzero().flatten().tolist())
            report(f"instance prompt {k + 1} reads only itself and its instance's source tokens", keys <= own)
    return ok


def check_tokens(args) -> bool:
    from transformers import Qwen3VLProcessor

    from bench import Bench
    from qwen21 import MODEL_ID, QwenImage21Pipeline
    from rosie.instance_edit import prepare
    from rosie.tokens import Prompter

    proc = Qwen3VLProcessor.from_pretrained(MODEL_ID, subfolder="processor")
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=proc, transformer=None)
    P = Prompter.from_pipe(pipe)
    bench = Bench(args.dataset_root)
    ok = True
    for sid in args.sample_ids:
        s = bench[sid]
        for mode in ("text", "image"):
            prep = prepare(P, s, local_mode=mode)
            L = prep["layout"]
            print(f"{sid} [{mode}] {prep['w']}x{prep['h']} {L.summary()}")
            exact, counts = groups_rebuild(allowed_mask(L), L.H * L.W)
            ok &= exact
            print(f"   {'ok  ' if exact else 'FAIL'} attention groups rebuild the mask: {counts[0]} groups at step 0, "
                  f"{counts[1]} on cached steps")
            for k, seg in enumerate(prep["locs"]):
                words = proc.tokenizer.decode(seg.ids[seg.tgt_words]).strip()
                good = words == s["targets"][k] and len(L.tgt_words[k]) == len(seg.tgt_words)
                ok &= good
                if not good or k < 2:
                    print(f"   {'ok  ' if good else 'FAIL'} instance {k}: {words!r} (bench: {s['targets'][k]!r}), "
                          f"{len(L.inst[k])} grid tokens")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", action="store_true")
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_ids", type=str, nargs="+", default=["06", "09", "276"])
    args = p.parse_args()
    ok = check_tokens(args) if args.tokens else check_tiny()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

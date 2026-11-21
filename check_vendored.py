"""
Check that the vendored Qwen-Image-2.1 code computes exactly what the installed diffusers does.

    python check_vendored.py           CPU, no weights: a tiny random transformer, the installed class vs the
                                       vendored one with the same weights, on a joint sequence with a condition
                                       image; prefill + a cached step, and a run without the cache. Must be equal.
    python check_vendored.py --full    GPU: the installed QwenImage21Pipeline vs load_pipeline() on one bench image,
                                       same seed; saves both and prints the max pixel difference (should be 0).
"""
import argparse
import sys

import torch


def tiny_inputs(dim_ctx: int, ch: int, gen: torch.Generator):
    """One sample: 3 text tokens, a 4x4 condition image (4 VLM slots), 5 text tokens, a 4x4 target (4 slots)."""
    pre, post, hc, wc, ht, wt = 3, 5, 4, 4, 4, 4
    cond_slots, tgt_slots = hc * wc // 4, ht * wt // 4
    vlm = [False] * pre + [True] * cond_slots + [False] * post
    img_mask = torch.tensor([vlm + [True] * tgt_slots])
    ehs = torch.randn(1, len(vlm), dim_ctx, generator=gen)
    ehs_mask = torch.ones(1, len(vlm), dtype=torch.long)
    hidden = torch.randn(1, hc * wc + ht * wt, ch, generator=gen)
    return dict(hidden_states=hidden, encoder_hidden_states=ehs, encoder_hidden_states_mask=ehs_mask,
                img_shapes=[[(1, hc, wc), (1, ht, wt)]], img_mask=img_mask)


def check_tiny() -> bool:
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache as RefCache, QwenImage21Transformer2DModel as Ref)
    from qwen21.transformer_qwenimage21 import QwenImage21KVCache as VenCache, QwenImage21Transformer2DModel as Ven

    assert Ref is not Ven
    cfg = dict(num_layers=2, attention_head_dim=16, num_attention_heads=2, context_in_dim=32, in_channels=8,
               out_channels=8, axes_dims_rope=(4, 6, 6))
    torch.manual_seed(0)
    ref = Ref(**cfg).eval()
    ven = Ven(**cfg).eval()
    for p in ref.parameters():  # zero-init layers (modulation, gates) would hide differences
        torch.nn.init.normal_(p, std=0.2)
    ven.load_state_dict(ref.state_dict())

    x = tiny_inputs(32, 8, torch.Generator().manual_seed(1))
    ok, outs = True, {}
    with torch.no_grad():
        for name, steps in [("no cache", [(0.9, None), (0.5, None)]), ("kv cache", [(0.9, "extract"), (0.5, "cached")])]:
            caches = (RefCache(2), VenCache(2)) if steps[0][1] else (None, None)
            for t, mode in steps:
                kw = dict(x, timestep=torch.tensor([t]), kv_cache_mode=mode, return_dict=False)
                a = ref(**kw, kv_cache=caches[0])[0]
                b = ven(**kw, kv_cache=caches[1])[0]
                same = torch.equal(a, b)
                ok &= same
                outs[name, t] = b
                print(f"{name:8s} t={t} mode={mode}: shape {tuple(a.shape)}, equal={same}, "
                      f"max|diff|={(a - b).abs().max().item():.3g}, |out|={a.abs().mean().item():.3g}")
        # the prefix (text + source) does not depend on the step, so a cached step equals recomputing everything:
        # what is read from the prefix at step 0 (the steering anchors) holds for every later step
        full, cached = outs["no cache", 0.5][:, -cached_len(x):], outs["kv cache", 0.5]
        close = torch.allclose(full, cached, atol=1e-5)
        ok &= close
        print(f"cached step == full recompute (target tokens): {close}, max|diff|={(full - cached).abs().max():.3g}")
    return ok


def cached_len(x) -> int:
    _, h, w = x["img_shapes"][0][-1]
    return h * w


def check_full(args) -> bool:
    import numpy as np
    from diffusers import QwenImage21Pipeline as Official

    from bench import Bench
    from qwen21 import MODEL_ID, load_pipeline

    s = Bench(args.dataset_root)[args.sample_id]
    call = dict(prompt=s["edit_inst_single"], image=s["image"], generator=None)
    out = {}
    for name in ("official", "vendored"):
        if name == "official":
            pipe = Official.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload(device=args.device) if args.cpu_offload else pipe.to(args.device)
        else:
            pipe = load_pipeline(MODEL_ID, args.device, cpu_offload=args.cpu_offload)
        print(name, type(pipe.transformer).__module__)
        call["generator"] = torch.Generator("cpu").manual_seed(0)
        out[name] = pipe(**call).images[0]
        out[name].save(f"check_{name}_{args.sample_id}.png")
        del pipe
        torch.cuda.empty_cache()
    d = np.abs(np.asarray(out["official"], np.int16) - np.asarray(out["vendored"], np.int16)).max()
    print(f"max pixel difference: {d}  (check_official_{args.sample_id}.png, check_vendored_{args.sample_id}.png)")
    return d == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--sample_id", type=str, default="06")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true", help="for GPUs under ~40 GB")
    args = p.parse_args()
    ok = check_full(args) if args.full else check_tiny()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

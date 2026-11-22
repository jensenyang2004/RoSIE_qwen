"""
Checks for the rotation steering (rosie/steer.py). CPU, no weights.

    python check_steer.py [--mice ../mice]
        1. the port: rotate() == mice's ResidualRotate._edit (flux2/residual_rotate.py, read from --mice; skipped if
           it is not there) on random updates, with each term alone and all four together
        2. a tiny random transformer on check_binding.py's synthetic layout, binding on:
           a. all strengths 0: the hooks leave the output unchanged
           b. steering on: the KV-cached run (anchors from the step-0 prefill) == recomputing everything every step
              (anchors captured again in every step), so the cached anchors are the ones a full pass would use
           c. each term alone changes the output and turns the tokens it should (stats > 0)
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from check_binding import inputs, run, synthetic_layout, tiny_model
from rosie.binding import allowed_mask, binding
from rosie.steer import RotationSteer, bg_weights, pair_weights, rotate, source_direction, steering

STRENGTHS = dict(ctc=0.9, src=0.25, bg=0.35, ins=0.2)


def check_port(mice: Path, report) -> None:
    if not (mice / "flux2" / "residual_rotate.py").exists():
        print(f"skip the port check: no {mice}/flux2/residual_rotate.py")
        return
    sys.path.insert(0, str(mice))
    from flux2.residual_rotate import ResidualRotate

    torch.manual_seed(0)
    H, W, Dm = 6, 7, 32
    masks = torch.zeros(3, H, W)
    masks[0, :3, :3] = 1
    masks[1, :3, 3:5] = 1  # touches instance 0
    masks[2, 4:, 4:] = 1
    flat = masks.reshape(3, -1) > 0.5
    h_in, h_out = torch.randn(1, H * W, Dm), torch.randn(1, H * W, Dm)
    D_src = torch.randn(1, H * W, Dm)
    U = F.normalize(torch.randn(1, 3, Dm), dim=-1)
    pw, bw = pair_weights(flat, H, W, 2.0), bg_weights(flat, H, W, 1.0, 2.0)
    for name, on in [(t, {t}) for t in STRENGTHS] + [("all four", set(STRENGTHS))]:
        st = {t: (v if t in on else 0.0) for t, v in STRENGTHS.items()}
        ref = ResidualRotate(rot_ctc=st["ctc"], rot_src=st["src"], rot_bg=st["bg"], rot_ins=st["ins"],
                             rot_ins_double=0.0, local_tau=2.0, bg_margin=1.0, bg_tau=2.0, sigma_terms=())
        ref.setup(list(masks), device="cpu")
        ref.sigma, ref.step_idx, ref.factor = 1.0, 0, 1.0
        want = ref._edit(h_in, h_out, "single_0", "lat", src=(D_src, D_src), sem=U)
        got, _ = rotate(h_in, h_out, flat, pw, bw, source_direction(D_src), U, st["ctc"], st["src"], st["bg"],
                        st["ins"], math.pi / 2)
        d = (got - want).abs().max().item()
        report(f"port == mice ResidualRotate ({name})", d < 1e-5 and (got - h_out).abs().max() > 0, f"max|diff|={d:.2g}")


def steered_run(model, L, x, steps, cache, steer):
    with binding(model, allowed_mask(L), L.H * L.W):
        if steer is None:
            return run(model, x, steps, cache)
        with steering(model, steer, L, steps):
            return run(model, x, steps, cache)


def check_tiny(report) -> None:
    L = synthetic_layout()
    T = L.H * L.W
    x = inputs(L, torch.Generator().manual_seed(1))
    steps = [0.9, 0.5, 0.3]
    every = dict(ctc_steps=None, src_steps=None, bg_steps=None, ins_steps=None, bg_margin=0.0, bg_tau=1.0)
    with torch.no_grad():
        model = tiny_model(3)
        ref = steered_run(model, L, x, steps, True, None)
        zero = RotationSteer(rot_ctc=0, rot_src=0, rot_bg=0, rot_ins=0, **every)
        got = steered_run(model, L, x, steps, True, zero)
        d = max((a - b).abs().max().item() for a, b in zip(ref, got))
        report("all strengths 0 == no steering", d == 0, f"max|diff|={d:.2g}")

        steer = RotationSteer(**every)
        cached = steered_run(model, L, x, steps, True, steer)
        n_stats = len(steer.stats)
        full = steered_run(model, L, x, steps, False, steer)
        d = max((a[:, -T:] - b[:, -T:]).abs().max().item() for a, b in zip(full, cached))
        moved = max((a - b).abs().max().item() for a, b in zip(cached, ref))
        report("steering on: cached anchors == recompute", d < 1e-5, f"max|diff|={d:.2g} (steering moved the "
               f"output by {moved:.2g}, {n_stats} block edits)")

        for t in STRENGTHS:
            st = {f"rot_{u}": (v if u == t else 0.0) for u, v in STRENGTHS.items()}
            steer = RotationSteer(**st, **every)
            out = steered_run(model, L, x, steps, True, steer)
            moved = max((a - b).abs().max().item() for a, b in zip(out, ref))
            deg = max(s.get(f"{t}_deg", 0.0) for s in steer.stats)
            report(f"{t} alone moves the output and turns its tokens", moved > 0 and deg > 0,
                   f"(output {moved:.2g}, up to {deg:.1f} deg)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mice", type=Path, default=Path("../mice"), help="the mice repo, for the port check")
    args = p.parse_args()
    ok = True

    def report(name, cond, extra=""):
        global ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {name} {extra}")

    check_port(args.mice, report)
    check_tiny(report)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

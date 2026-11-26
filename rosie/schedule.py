"""
Step count other than Turbo's 8. The pipeline ignores num_inference_steps when the model has sample_sigmas (Turbo
does), so a different count has to be passed as sigmas. The scheduler has no shift (use_dynamic_shifting false,
shift 1.0), so the sigmas given are the sigmas the model gets.
"""
import numpy as np


def resample_sigmas(base: list, n: int) -> list:
    """n sigmas on the curve of the base schedule: base[i] at step fraction i/K, the terminal 0 at 1, linear in
    between. n = K gives base back; n = 2K keeps every base sigma and adds the midpoints."""
    K = len(base)
    if n < 1:
        raise ValueError(f"--steps {n}: need at least 1")
    if n == K:
        return [float(s) for s in base]
    u = np.arange(K + 1) / K
    s = np.append(np.asarray(base, dtype=np.float64), 0.0)
    return [float(v) for v in np.interp(np.arange(n) / n, u, s)]


def remap_steps(idx: tuple | None, base: list, new: list) -> tuple | None:
    """A set of base steps -> the new steps over the same noise range. Base step i takes the latent from base[i] to
    base[i + 1], so the base steps lo..hi cover (base[hi + 1], base[lo]], and new step j is in it if new[j] is.
    None (every step) and () (none) stay."""
    if idx is None or idx == () or len(new) == len(base):
        return idx
    lo, hi = min(idx), max(idx)
    if sorted(idx) != list(range(lo, hi + 1)):
        raise ValueError(f"steps {idx}: only a contiguous range can be moved to another step count")
    top = base[lo]
    bottom = base[hi + 1] if hi + 1 < len(base) else 0.0
    return tuple(j for j, s in enumerate(new) if bottom + 1e-6 < s <= top + 1e-6)

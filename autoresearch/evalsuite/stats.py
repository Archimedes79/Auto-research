"""Statistics of the suite: robust outlier scores, leave-one-out influence, paired scene bootstrap. numpy only.

    robust_z(values)                      median / MAD z-scores (a few bad frames do not move their own yardstick)
    iqr_fences(values, k)                 Tukey fences
    loo_influence(units, reduce, keys)    how much each unit (scene or frame) moves the dataset-level numbers: all - without it
    bootstrap_scenes(a, b, reduce, ...)   paired bootstrap over scenes of the difference of two runs' numbers
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

MAD_TO_SIGMA = 1.4826


def robust_z(values: Sequence[float]) -> np.ndarray:
    """(v - median) / (1.4826 MAD); nan stays nan. MAD = 0 (more than half the values equal): the mean absolute deviation
    scaled to sigma is used, and if that is 0 as well every score is 0."""
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return out
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med)) * MAD_TO_SIGMA
    if mad <= 0:
        mad = np.mean(np.abs(v[ok] - med)) * 1.2533
    out[ok] = 0.0 if mad <= 0 else (v[ok] - med) / mad
    return out


def iqr_fences(values: Sequence[float], k: float = 1.5):
    """(low, high) Tukey fences of the finite values"""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def loo_influence(units: Dict[str, list], reduce: Callable[[list], Dict[str, float]], keys: Optional[Sequence[str]] = None
                  ) -> Dict[str, Dict[str, float]]:
    """units: name -> list of per-frame Counts of that unit (a scene, or one frame). reduce(list of Counts) -> numbers.
    -> name -> {key: all - without_unit}. A positive value of a high-is-better number means the unit pulls the number UP
    (the number is higher with it than without it). Exact, because the numbers are reductions of additive counts."""
    names = list(units)
    allc = [c for n in names for c in units[n]]
    full = reduce(allc)
    keys = list(keys) if keys is not None else list(full)
    out = {}
    for n in names:
        rest = [c for m in names if m != n for c in units[m]]
        wo = reduce(rest) if rest else {k: float("nan") for k in keys}
        out[n] = {k: _sub(full.get(k, float("nan")), wo.get(k, float("nan"))) for k in keys}
    return out


def _sub(a: float, b: float) -> float:
    return float(a) - float(b) if (a == a and b == b) else float("nan")


def bootstrap_scenes(a: Dict[str, list], b: Dict[str, list], reduce: Callable[..., Dict[str, float]], keys: Sequence[str],
                     n: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap over scenes: a, b: scene -> list of Counts (two runs on the same scenes; only the scenes in both are
    used). Each draw samples scenes with replacement (the same draw for both), reduce() gives both runs' numbers, the
    difference b - a is recorded. -> key -> {a, b, diff, lo, hi, p_b_better_or_equal (share of draws with b >= a), n_scenes}.
    reduce(counts, keys=keys) must accept the keys argument."""
    scenes = sorted(set(a) & set(b))
    if not scenes:
        raise ValueError("the two runs have no scene in common")
    rng = np.random.default_rng(seed)
    ra, rb = reduce([c for s in scenes for c in a[s]], keys=keys), reduce([c for s in scenes for c in b[s]], keys=keys)
    draws = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.integers(0, len(scenes), len(scenes))
        sa = reduce([c for i in pick for c in a[scenes[i]]], keys=keys)
        sb = reduce([c for i in pick for c in b[scenes[i]]], keys=keys)
        for k in keys:
            draws[k].append(sb[k] - sa[k])
    out = {}
    for k in keys:
        d = np.asarray(draws[k], float)
        ok = np.isfinite(d)
        lo, hi = (np.percentile(d[ok], [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.any() else (float("nan"),) * 2)
        out[k] = {"a": float(ra[k]), "b": float(rb[k]), "diff": float(rb[k] - ra[k]), "lo": float(lo), "hi": float(hi),
                  "p_b_ge_a": float((d[ok] >= 0).mean()) if ok.any() else float("nan"), "n_scenes": len(scenes)}
    return out

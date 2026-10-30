"""Looking into a network: generic pictures of tapped tensors (by kind), their statistics and plausibility flags.

A Subject declares `TapSpec`s (name, kind, stage, what it should look like, what a problem looks like) and returns the arrays of
the taps it was asked for. This module turns any such array into panels without knowing the network:
    feature_map (C, H, W)   energy (L2 norm over channels) and the first three principal components as colours
    map (H, W)              the field itself      image (H, W[, 3])   the picture      logits (K, H, W)   max sigmoid over K
    vector (N,) / scalar    bars / text
and `plausibility` flags what is wrong without looking: non-finite values, a dead (all-zero or constant) tensor, no spatial
structure, an extreme magnitude. Flags are heuristics: they point, a human judges.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..contracts import Flag, Panel, TapSpec


def energy(t) -> np.ndarray:
    """(C, H, W) -> (H, W): L2 norm over the channels"""
    return np.sqrt((np.asarray(t, np.float64) ** 2).sum(0)).astype(np.float32)


def pca_rgb(t) -> np.ndarray:
    """(C, H, W) -> (H, W, 3): first three principal components of the per-cell feature vectors, scaled to [0, 1]"""
    C, H, W = t.shape
    f = np.asarray(t, np.float64).reshape(C, -1).T
    f = f - f.mean(0)
    k = min(3, C, f.shape[0])
    _, _, vt = np.linalg.svd(f[:: max(1, len(f) // 4000)], full_matrices=False)
    p = np.zeros((f.shape[0], 3))
    p[:, :k] = f @ vt[:k].T
    lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
    return np.clip((p - lo) / np.maximum(hi - lo, 1e-9), 0, 1).reshape(H, W, 3).astype(np.float32)


def stats(name: str, arr) -> Dict[str, object]:
    a = np.asarray(arr, np.float64)
    fin = np.isfinite(a)
    b = a[fin] if fin.any() else np.zeros(1)
    return {"tap": name, "shape": "x".join(str(s) for s in a.shape), "mean": float(b.mean()), "std": float(b.std()),
            "abs_max": float(np.abs(b).max()), "nonzero": float((a != 0).mean()), "finite": bool(fin.all())}


def plausibility(spec: TapSpec, arr) -> List[Flag]:
    a = np.asarray(arr, np.float64)
    out = []
    if not np.isfinite(a).all():
        out.append(Flag(f"{spec.name}: finite", "fail", float((~np.isfinite(a)).mean()), "NaN or Inf in the tensor"))
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size and (a == 0).all():
        out.append(Flag(f"{spec.name}: alive", "fail", 0.0, "all zero (dead stage or an input that never arrived)"))
    elif a.size and a.std() < 1e-9:
        out.append(Flag(f"{spec.name}: alive", "warn", float(a.std()), "constant tensor"))
    if a.ndim == 3 and a.size:
        e = energy(a)
        if e.size and e.std() < 1e-6 * max(float(e.mean()), 1e-9) and e.max() > 0:
            out.append(Flag(f"{spec.name}: structure", "warn", float(e.std()), "same energy everywhere (no spatial structure)"))
    if a.size and np.abs(a).max() > 1e6:
        out.append(Flag(f"{spec.name}: magnitude", "warn", float(np.abs(a).max()), "extreme values"))
    return out


def _limits(spec: TapSpec):
    return ((spec.extent[1], spec.extent[0]) if (spec.flip_x and spec.extent) else None), None


def tap_panels(taps: Dict[str, np.ndarray], specs: Sequence[TapSpec], wanted: Optional[Sequence[str]] = None,
               pca: bool = True) -> List[Panel]:
    """panels for the tapped arrays, in the order the specs were declared (`wanted`: only these names)."""
    by = {s.name: s for s in specs}
    order = [s.name for s in specs if s.name in taps] + [n for n in taps if n not in by]
    panels: List[Panel] = []
    for name in order:
        if wanted is not None and name not in wanted:
            continue
        spec = by.get(name, TapSpec(name=name, kind="feature_map" if np.asarray(taps[name]).ndim == 3 else "map"))
        a = np.asarray(taps[name])
        stage = f"[{spec.stage}] " if spec.stage else ""
        xlim, ylim = _limits(spec)
        kw = dict(extent=spec.extent, xlim=xlim, ylim=ylim, note=spec.note)
        if spec.kind == "feature_map" and a.ndim == 3:
            panels.append(Panel("raster", f"{stage}{name}: energy {a.shape[0]}ch", energy(a), colorbar=True, **kw))
            if pca and a.shape[0] >= 3:
                panels.append(Panel("rgb", f"{stage}{name}: first 3 components", pca_rgb(a), **kw))
        elif spec.kind == "logits" and a.ndim == 3:
            panels.append(Panel("raster", f"{stage}{name}: max sigmoid over {a.shape[0]}ch", (1 / (1 + np.exp(-a))).max(0),
                                colorbar=True, vmin=0, vmax=1, **kw))
        elif spec.kind == "image" or (a.ndim == 3 and a.shape[-1] in (1, 3) and spec.kind != "feature_map"):
            panels.append(Panel("image", f"{stage}{name}", a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a, note=spec.note))
        elif a.ndim == 2:
            panels.append(Panel("raster", f"{stage}{name}", a, colorbar=True, **kw))
        elif a.ndim <= 1 and a.size > 1:
            panels.append(Panel("bars", f"{stage}{name}", {str(i): float(v) for i, v in enumerate(a.ravel()[:16])}, note=spec.note))
        else:
            panels.append(Panel("text", f"{stage}{name}", f"{a.ravel()[:8].tolist()}", note=spec.note))
    return panels


def stats_table(taps: Dict[str, np.ndarray], flags: Sequence[Flag] = ()) -> Panel:
    rows = [["tap", "shape", "mean", "std", "abs_max", "nonzero", "finite"]]
    for n, a in taps.items():
        s = stats(n, a)
        rows.append([n, s["shape"], s["mean"], s["std"], s["abs_max"], s["nonzero"], s["finite"]])
    bad = [f for f in flags if f.status != "ok"]
    if bad:
        rows.append([])
        rows.append(["flags"])
        rows += [[f.status.upper(), f.name, f.note] for f in bad]
    return Panel("table", "tap statistics and plausibility flags", rows)

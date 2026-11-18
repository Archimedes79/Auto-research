"""Route input of one frame for the joint route + builder fine-tune (loop 17 / 18): the resampled item of an assembled graph and the interpolation matrix M [n_new, n_old]
(the route tokens are the builder's node tokens gathered at the surviving vertices and interpolated along the edges: a fixed sparse matrix, so a route loss reaches the builder).
numpy, but it imports p16_tokens / p16_tok_resample, which load torch and the p16 modules as a side effect (no data is loaded at import). Used by training/train/p17_joint.py, p18_joint.py, evaluation/loops/loop17/p17_consumer.py and evaluation/drivers/e2e_gradient_report.py
(the functions were identical copies in p17_joint.py and p18_joint.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_tok_resample as TR  # noqa: E402
import p16_tokens as TK  # noqa: E402


def resample_matrix(it, step):
    """the interpolation of p16_tok_resample.resample_item as a matrix [n_new, n_old] (same order of the new points)"""
    xy = it["xy"].astype(np.float64)
    has = it["has"].astype(bool)
    n = len(xy)
    rows = [({i: 1.0} if has[i] else {}) for i in range(n)]
    seen = set()
    for path in it["pieces"]:
        for a, b in zip(path[:-1], path[1:]):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            lo, hi = key
            m = int(np.ceil(float(np.linalg.norm(xy[hi] - xy[lo])) / step)) - 1
            for j in range(1, m + 1):
                w = j / (m + 1)
                if has[lo] and has[hi]:
                    rows.append({lo: 1 - w, hi: w})
                elif has[lo]:
                    rows.append({lo: 1.0})
                elif has[hi]:
                    rows.append({hi: 1.0})
                else:
                    rows.append({})
    M = np.zeros((len(rows), n), np.float32)
    for r, d in enumerate(rows):
        for c, w in d.items():
            M[r, c] = w
    return M


def route_item(r, ax, g, step=4.0):
    """structure of the route input of one frame (numpy): the resampled item and the interpolation matrix; tokens are added by the caller (differentiably)"""
    V = np.asarray(ax["V"], np.float32).reshape(-1, 2)
    src = np.asarray(ax["src"], int)
    has = src >= 0
    keep = np.where(has, r["pi"][np.clip(src, 0, None)], 0.5).astype(np.float32)
    it0 = dict(xy=V, tok=np.zeros((len(V), 128), np.float16), has=has, cp=TK.cross_section(V, ax["edges"]), pieces=[list(map(int, p)) for p in g["nodes"]], edges=sorted((int(x), int(y)) for x, y in ax["edges"]), keep=keep)
    it1 = TR.resample_item(it0, step)
    M = resample_matrix(it0, step)
    assert len(M) == len(it1["xy"])
    return it0, it1, M, src, has

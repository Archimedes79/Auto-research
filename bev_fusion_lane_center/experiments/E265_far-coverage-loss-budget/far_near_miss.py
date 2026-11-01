"""E265 step 0b (card author's pre-measurement, CPU only): what are the far misses of the output graph that lie 1 - 2 m from it?
For every far GT point (>= 40 m, dense 0.5 m, |x|,|y| < 48): distance to the output, the radial / lateral components of the offset to the
nearest output point, the GT turning over +-5 m, and the length of the output vertex-to-vertex segment that is nearest.
Usage: tools/env/_py14.sh <this file> <registry name | control> [n_frames]"""
import sys

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV
import graph_eval as GE

name = sys.argv[1] if len(sys.argv) > 1 else "recall"
nf = int(sys.argv[2]) if len(sys.argv) > 2 else None
reg = EV.registry()
g, gts, info = EV.graphs_of(name, reg[name], "val", require_cache=True)
if nf:
    g, gts = g[:nf], gts[:nf]
rows = []  # d, radial error (pred - gt range, m), lateral |offset| across the GT tangent, turning deg over +-5 m, segment length
for pred, gt in zip(g, gts):
    G, DG, IG = GE.dense_points(gt)
    far = np.linalg.norm(G, axis=1) >= 40.0
    if not far.any():
        continue
    P, _, IP = GE.dense_points(pred)
    if not len(P):
        rows += [(np.inf, np.nan, np.nan, np.nan, np.nan)] * int(far.sum())
        continue
    d, j = cKDTree(P).query(G[far])
    # vertex-to-vertex segment lengths of the prediction: the polyline points of piece IP[j] (pieces carry the vertex positions)
    seg_len = []
    for jj in j:
        pl = np.asarray(pred["pts"][IP[jj]], float)
        k = int(np.argmin(np.linalg.norm(pl - P[jj], axis=1)))
        nb = [np.linalg.norm(pl[k] - pl[k - 1]) if k > 0 else 0.0, np.linalg.norm(pl[k + 1] - pl[k]) if k + 1 < len(pl) else 0.0]
        seg_len.append(max(nb))
    gi = np.nonzero(far)[0]
    for t, (dd, jj, sl) in enumerate(zip(d, j, seg_len)):
        k = gi[t]
        off = P[jj] - G[k]
        rad = float(np.linalg.norm(P[jj]) - np.linalg.norm(G[k]))
        n = np.array([-DG[k, 1], DG[k, 0]])
        lat = abs(float(off @ n))
        same = np.nonzero(IG == IG[k])[0]
        a, b = max(same[0], k - 10), min(same[-1], k + 10)
        c = float(np.clip(DG[a] @ DG[b], -1, 1))
        rows.append((dd, rad, lat, np.degrees(np.arccos(c)), sl))
R = np.array(rows, float)
d = R[:, 0]
print(f"{name}: far GT points {len(R)}, R@1 {np.mean(d <= 1):.4f}, R@1.5 {np.mean(d <= 1.5):.4f}, R@2 {np.mean(d <= 2):.4f}, R@3 {np.mean(d <= 3):.4f}")
for nm, m in (("hit <= 1 m", d <= 1), ("near miss 1 - 2 m", (d > 1) & (d <= 2)), ("miss > 2 m", d > 2)):
    X = R[m]
    if not len(X):
        continue
    print(f"  {nm:18s} share {m.mean():.4f}  radial error median {np.nanmedian(X[:, 1]):+.2f} m (pred closer: {np.nanmean(X[:, 1] < 0):.2f})  lateral median {np.nanmedian(X[:, 2]):.2f} m"
          f"  GT turning over 10 m: median {np.nanmedian(X[:, 3]):.1f} deg, > 10 deg {np.nanmean(X[:, 3] > 10):.2f}  nearest segment median {np.nanmedian(X[:, 4]):.1f} m, > 12 m {np.nanmean(X[:, 4] > 12):.2f}")

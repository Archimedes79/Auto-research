"""scratch (continuation of target_vs_pred.py, Mirko's remark of 2026-11-08): where does the map deviate from the target before the junction - a lateral shift or a width, and does it depend on the junction or on the range from the ego?
Reads temp/e284/target_vs_pred.npz (the stored p and T of the 88 frames + the picture frame). For every GT lane point (every 0.5 m, the g05 of the training target, |x|, |y| < 48): the profile of p and of T across the lane
(bilinear samples at -1.7 ... +1.7 m along the normal): the centroid (lateral shift of the map against the lane, metres, + = left of the travel direction) and the width (weighted standard deviation).
Grouped by the range of the point from the ego (0-16, 16-32, 32-48 m) and by its distance to the junction (the connector points of the lane GT inside an intersection): < 8 m, 8-24 m, >= 24 m / none.
Points near another lane (a GT point of ANOTHER piece within 3 m, or within the junction) are left out: their profile mixes lanes.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/lateral_offset.py"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks = [str(t) for t in z["tokens"]]
P, T = z["p"].astype(np.float32).reshape(-1, 100, 100), z["t"].astype(np.float32).reshape(-1, 100, 100)
by_tok = {f.token: f for f in QE.val_frames()}
S = np.arange(-1.7, 1.71, 0.17)                                   # 21 samples across the lane
rows = []                                                         # per point: range, dj, bias_p, bias_t, width_p, width_t, alone, bx, by (shift in x / y, metres)
for i, tk in enumerate(toks):
    f = by_tok[tk]
    g = f.lane()
    nt = f.ref().nt
    sp = nt["pts"][nt["supp"]]
    jt = cKDTree(sp) if len(sp) else None
    pcs = [np.asarray(p["pts"], float).reshape(-1, 2) for p in g.get("pieces", []) if len(p["pts"]) >= 2]
    allp = np.concatenate(pcs)
    owner = np.concatenate([np.full(len(q), k) for k, q in enumerate(pcs)])
    tan = np.concatenate([QT._tangents(q) for q in pcs])
    keep = (np.abs(allp) < 46.0).all(1)
    allp, owner, tan = allp[keep], owner[keep], tan[keep]
    tree = cKDTree(allp)
    dj = jt.query(allp)[0] if jt is not None else np.full(len(allp), 1e9)
    nb = tree.query_ball_point(allp, 3.0)
    alone = np.array([len(set(owner[j])) == 1 for j in nb])
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)                    # left of the travel direction
    for name, M in (("p", P[i]), ("t", T[i])):
        xy = allp[:, None, :] + S[None, :, None] * nrm[:, None, :]                 # (N, 21, 2)
        v = map_coordinates(M, [(xy[..., 0] + 49.5).ravel(), (xy[..., 1] + 49.5).ravel()], order=1, mode="nearest").reshape(len(allp), len(S))
        w = np.clip(v, 0, None)
        sw = w.sum(1) + 1e-9
        cen = (w * S[None]).sum(1) / sw
        wid = np.sqrt((w * (S[None] - cen[:, None]) ** 2).sum(1) / sw)
        if name == "p":
            cp, wp, mp = cen, wid, v.max(1)
        else:
            ct, wt, mt = cen, wid, v.max(1)
    rng = np.linalg.norm(allp, axis=1)
    for k in range(len(allp)):
        rows.append((rng[k], dj[k], cp[k], ct[k], wp[k], wt[k], float(alone[k]), mp[k], mt[k], nrm[k, 0], nrm[k, 1], allp[k, 0], allp[k, 1]))
R = np.array(rows)
rng, dj, cp, ct, wp, wt, alone, mp, mt = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4], R[:, 5], R[:, 6] > 0.5, R[:, 7], R[:, 8]
shift = cp - ct                                                    # shift of the map against the target ridge (the target ridge is the lane up to the neighbours' pull)
out = {}
RB = ((0, 16), (16, 32), (32, 48))
JB = (("< 8 m", 0, 8), ("8-24 m", 8, 24), (">= 24 m or no junction", 24, 1e12))
for rb in RB:
    for jn, j0, j1 in JB:
        m = alone & (rng >= rb[0]) & (rng < rb[1]) & (dj >= j0) & (dj < j1) & (mt > 0.5)
        if m.sum() < 30:
            continue
        out[f"range {rb[0]}-{rb[1]} m | junction {jn}"] = dict(points=int(m.sum()), shift_median=float(np.median(shift[m])), shift_abs_median=float(np.median(np.abs(shift[m]))), shift_abs_p90=float(np.percentile(np.abs(shift[m]), 90)),
                                                              width_map=float(np.median(wp[m])), width_target=float(np.median(wt[m])), peak_map_median=float(np.median(mp[m])), share_peak_below_0_5=float((mp[m] < 0.5).mean()))
json.dump(out, open(os.path.join(ROOT, "temp", "e284", "lateral_offset.json"), "w"), indent=1)
print("points total", len(R), "alone", int(alone.sum()))
for k, v in out.items():
    print(f"{k:46s} n {v['points']:6d}  |shift| median {v['shift_abs_median']:.2f} m  p90 {v['shift_abs_p90']:.2f} m  (signed median {v['shift_median']:+.2f})  width map {v['width_map']:.2f} vs target {v['width_target']:.2f}  peak median {v['peak_map_median']:.2f}  peak < 0.5 in {100 * v['share_peak_below_0_5']:.0f} %")
# the same by range only, and by junction only (so the two effects can be told apart)
for rb in RB:
    m = alone & (rng >= rb[0]) & (rng < rb[1]) & (mt > 0.5)
    print(f"range {rb[0]}-{rb[1]} m, all junction distances: n {int(m.sum()):6d}  |shift| median {np.median(np.abs(shift[m])):.2f}  p90 {np.percentile(np.abs(shift[m]), 90):.2f}  width {np.median(wp[m]):.2f} vs {np.median(wt[m]):.2f}")
for jn, j0, j1 in JB:
    m = alone & (dj >= j0) & (dj < j1) & (mt > 0.5)
    print(f"junction {jn:24s}, all ranges: n {int(m.sum()):6d}  |shift| median {np.median(np.abs(shift[m])):.2f}  p90 {np.percentile(np.abs(shift[m]), 90):.2f}  width {np.median(wp[m]):.2f} vs {np.median(wt[m]):.2f}")

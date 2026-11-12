"""scratch (continuation of entry_and_blur.py, Mirko's marks of 2026-11-08): the lane points up to the junction are clean (median shift 0.13 m), so the misfit must start where the CONNECTORS start.
 1. CONNECTOR points by arc length from the start of their connector run (0-3, 3-6, 6-10, 10-15, 15-25 m): the shift of p's cross-lane centroid against T's (+ = left of the travel direction), |shift|, the share of points where the peak of p
    on the profile is < 0.5 although T's is > 0.5 (the connector is missing in the map), and the peak of p, T.
 2. the blur by the complexity of the junction: per frame the number of connector segments inside an intersection (>= 4 points; as gt_connectors.py), against the correlation of p and T and the gradient ratio over the cells within 8 m of
    the junction points; frames grouped 1-4, 5-10, 11-20, over 20 connectors.
Reads temp/e284/target_vs_pred.npz. CPU.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/conn_start_and_complexity.py"""
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
P = z["p"].astype(np.float32).reshape(-1, 100, 100)
T = z["t"].astype(np.float32).reshape(-1, 100, 100)
DJ = z["dj"].astype(np.float32).reshape(-1, 100, 100)
by_tok = {f.token: f for f in QE.val_frames()}
S = np.arange(-1.7, 1.71, 0.17)
n_stat = len(toks) - (1 if len(toks) > 88 else 0)
ARC = (("0-3 m", 0, 3), ("3-6 m", 3, 6), ("6-10 m", 6, 10), ("10-15 m", 10, 15), ("15-25 m", 15, 25))


def profile(M, xy):
    v = map_coordinates(M, [(xy[..., 0] + 49.5).ravel(), (xy[..., 1] + 49.5).ravel()], order=1, mode="nearest").reshape(xy.shape[:2])
    w = np.clip(v, 0, None)
    sw = w.sum(1) + 1e-9
    return (w * S[None]).sum(1) / sw, v.max(1)


rows, per_frame = [], []
inb = np.zeros((100, 100), bool)
inb[2:98, 2:98] = True
for i in range(n_stat):
    f = by_tok[toks[i]]
    g = f.lane()
    n_conn = 0
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
                if s["in_intersection"] and s["i1"] - s["i0"] >= 4:
                    n_conn += 1
        if not conn.any():
            continue
        arc = np.zeros(len(q))
        run = 0.0
        for k in range(len(q)):
            if conn[k]:
                run = run + (0.5 if k > 0 and conn[k - 1] else 0.0)
                arc[k] = run
            else:
                run = 0.0
        ok = conn & (np.abs(q) < 46.0).all(1)
        if not ok.any():
            continue
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)[ok]
        qq = q[ok]
        xy = qq[:, None, :] + S[None, :, None] * nrm[:, None, :]
        cp, mp = profile(P[i], xy)
        ct, mt = profile(T[i], xy)
        for k in range(len(qq)):
            rows.append((arc[ok][k], cp[k] - ct[k], mp[k], mt[k]))
    p, t, dj = P[i], T[i], DJ[i]
    m = (dj < 8.0) & ((p > 0.05) | (t > 0.05)) & inb
    if n_conn and m.sum() > 50:
        gp, gt = np.hypot(*np.gradient(p)), np.hypot(*np.gradient(t))
        per_frame.append((n_conn, float(np.corrcoef(p[m], t[m])[0, 1]), float(gp[m].sum() / gt[m].sum()), float(p[m].mean()), float(t[m].mean())))
    f._lane = f._ref = None
R = np.array(rows)
arc, sh, mp, mt = R.T
res = {}
for name, a, b in ARC:
    m = (arc >= a) & (arc < b) & (mt > 0.5)
    if m.sum() < 30:
        continue
    res[name] = dict(points=int(m.sum()), shift_left_median=float(np.median(sh[m])), shift_left_mean=float(sh[m].mean()), abs_median=float(np.median(np.abs(sh[m]))), abs_p90=float(np.percentile(np.abs(sh[m]), 90)),
                     share_abs_over_0_5=float((np.abs(sh[m]) > 0.5).mean()), share_left_over_0_5=float((sh[m] > 0.5).mean()), share_right_over_0_5=float((sh[m] < -0.5).mean()), share_missing=float((mp[m] < 0.5).mean()), peak_p=float(np.median(mp[m])), peak_t=float(np.median(mt[m])))
print("CONNECTOR points by arc length from the start of the connector (+ = left of the travel direction)")
for k, v in res.items():
    print(f"  {k:8s} n {v['points']:6d}  median {v['shift_left_median']:+.2f}  mean {v['shift_left_mean']:+.2f}  |median| {v['abs_median']:.2f}  p90 {v['abs_p90']:.2f}  >0.5 m: {100 * v['share_abs_over_0_5']:.0f} % (left {100 * v['share_left_over_0_5']:.0f} %, right {100 * v['share_right_over_0_5']:.0f} %)  p peak < 0.5: {100 * v['share_missing']:.0f} %  peak p {v['peak_p']:.2f}  peak T {v['peak_t']:.2f}")
PF = np.array(per_frame)
cx = {}
for name, a, b in (("1-4 connectors", 1, 4), ("5-10", 5, 10), ("11-20", 11, 20), ("over 20", 21, 10 ** 6)):
    m = (PF[:, 0] >= a) & (PF[:, 0] <= b)
    if m.sum() < 3:
        continue
    cx[name] = dict(frames=int(m.sum()), corr_p_T=float(PF[m, 1].mean()), gradient_ratio=float(PF[m, 2].mean()), mean_p=float(PF[m, 3].mean()), mean_T=float(PF[m, 4].mean()))
print("JUNCTION cells (within 8 m of the junction points) by the number of connectors in the frame")
for k, v in cx.items():
    print(f"  {k:16s} frames {v['frames']:3d}  corr(p, T) {v['corr_p_T']:.2f}  gradient p / T {v['gradient_ratio']:.2f}  mean p {v['mean_p']:.2f}  mean T {v['mean_T']:.2f}")
json.dump(dict(connector_start=res, complexity=cx), open(os.path.join(ROOT, "temp", "e284", "conn_start_and_complexity.json"), "w"), indent=1)

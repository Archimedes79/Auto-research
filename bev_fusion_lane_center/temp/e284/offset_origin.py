"""scratch (continuation of err_taxonomy.py): where do the 1 - 2.5 m OFFSETS of the delivered graph come from (11.5 % of the GT length, 8.3 % even on plain road lanes), and why did ridge snapping (snap_precheck.py, 9 frames: mean move 0.2 m,
F1@1.0 +0.0003) not fix them? CPU, model of baseline B seed 0 (runs/e273_p_s0) on every 12th VAL frame (73), hard close-node step 1.5 m, read-out 0.5. For every delivered polyline point of the SECTION kind (0.5 m samples, |x|, |y| < 48):
dg = distance to the nearest GT lane point (lane GT, all pieces); the distance to the nearest kept node (along the polyline, from the vertices); the ridge of the centre map p along the local normal (centroid within +-1.0 m of the profile maximum over +-2.0 m): its displacement,
whether it is a real ridge (maximum >= 0.6 and >= 0.25 above the lowest end value), and the distance of the ridge point itself to the GT (dg_ridge). Tables:
 1. by distance to the nearest node (0-1, 1-2, 2-3, 3-5, > 5 m): the share of points with dg > 1.0 (offset) and the median dg
 2. for the offset points (1.0 < dg <= 2.5): the share with a real ridge, and among those the share whose ridge point is within 1.0 m of the GT (snapping would fix it) and the share where the ridge is also off;
    for the offset points without a real ridge: the median maximum of the profile
 3. zone: plain road (farther than 8 m from a GT connector point inside an intersection) against the junction zone
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e284/offset_origin.py [--every 12]"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qpredict as QP  # noqa: E402

EVERY = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 12
S = np.arange(-2.0, 2.01, 0.25)
SPEC = QR.parse("close:1.5@0.5")
fn = QP.make_pred_fn("e273_p_s0", "exact", "cpu", keep_maps=True)
frames = QE.val_frames()[::EVERY]
rows = []                                                                    # per point: dg, d_node, ridge_disp, ok, vmax, dg_ridge, junction_zone, e of the nearest node
t0 = time.time()
for n, f in enumerate(frames):
    pred = fn(f.token)
    pmap = np.asarray(pred["maps"]["centre1"][0], float)
    g = f.lane()
    P, J = [], []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        inter = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector" and s["in_intersection"]:
                inter[s["i0"]: s["i1"] + 1] = True
        P.append(q)
        J.append(inter)
    P, J = np.concatenate(P), np.concatenate(J)
    gt_tree = cKDTree(P)
    jt = cKDTree(P[J]) if J.any() else None
    q_ = QR.apply(pred, SPEC)
    graph, bk = QE.readout(q_, 0.5, 0.5)
    kp = pred["nodes"]["pos"][bk["keep"]]
    ke = pred["nodes"]["e"][bk["keep"]]
    ktree = cKDTree(kp) if len(kp) else None
    for i, line in enumerate(graph.lines):
        if bk["kind"][i] != "section":
            continue
        line = np.asarray(line, float)
        if len(line) < 5:
            continue
        tan = np.gradient(line, axis=0)
        for _ in range(2):
            tan = np.c_[np.convolve(tan[:, 0], np.ones(5) / 5, "same"), np.convolve(tan[:, 1], np.ones(5) / 5, "same")]
        tn = np.linalg.norm(tan, axis=1, keepdims=True)
        tan = tan / np.where(tn > 1e-9, tn, 1.0)
        nrm = np.c_[-tan[:, 1], tan[:, 0]]
        xy = line[:, None, :] + S[None, :, None] * nrm[:, None, :]
        v = map_coordinates(pmap, [(xy[..., 0] + 49.5).ravel(), (xy[..., 1] + 49.5).ravel()], order=1, mode="nearest").reshape(len(line), len(S))
        am, vmax = v.argmax(1), v.max(1)
        lo = np.minimum(v[:, 0], v[:, -1])
        ok = (vmax >= 0.6) & (vmax - lo >= 0.25)
        w = np.where(np.abs(S[None, :] - S[am][:, None]) <= 1.0, v, 0.0)
        disp = np.clip((w * S[None]).sum(1) / (w.sum(1) + 1e-9), -2.0, 2.0)
        ridge_pt = line + disp[:, None] * nrm
        dg = gt_tree.query(line)[0]
        dgr = gt_tree.query(ridge_pt)[0]
        dn, jn = ktree.query(line) if ktree is not None else (np.full(len(line), 1e9), np.zeros(len(line), int))
        zj = (jt.query(line)[0] <= 8.0) if jt is not None else np.zeros(len(line), bool)
        inb = (np.abs(line) < 48.0).all(1)
        for k in np.flatnonzero(inb):
            rows.append((dg[k], dn[k], disp[k], float(ok[k]), vmax[k], dgr[k], float(zj[k]), ke[jn[k]] if ktree is not None else 0.0))
    f._lane = f._ref = None
    if (n + 1) % 10 == 0:
        print(f"  frame {n + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
R = np.array(rows)
dg, dn, disp, ok, vmax, dgr, zj, en = R.T
ok = ok > 0.5
zj = zj > 0.5
res = dict(points=int(len(R)), frames=len(frames))
print(f"{len(R)} section points on {len(frames)} frames")
print("1. by the distance to the nearest kept node: share of points with dg > 1.0 m (offset), median dg")
res["by_node_distance"] = {}
for a, b in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)):
    m = (dn >= a) & (dn < b)
    res["by_node_distance"][f"{a}-{b}"] = dict(n=int(m.sum()), offset_share=float((dg[m] > 1.0).mean()), median_dg=float(np.median(dg[m])), p90_dg=float(np.percentile(dg[m], 90)))
    print(f"   {a:>2}-{b if b < 1e8 else 'inf':>3} m from a node: n {m.sum():7d}  offset share {100 * (dg[m] > 1.0).mean():5.1f} %  median dg {np.median(dg[m]):.2f} m  p90 {np.percentile(dg[m], 90):.2f} m")
off = (dg > 1.0) & (dg <= 2.5)
print(f"2. offset points (1.0 < dg <= 2.5): {off.sum()} = {100 * off.mean():.1f} % of the points")
for zname, zm in (("plain road", ~zj), ("junction zone", zj), ("all", np.ones(len(R), bool))):
    m = off & zm
    if not m.sum():
        continue
    real = m & ok
    fixable = real & (dgr <= 1.0)
    worse = real & (dgr > dg)
    res[f"offset_{zname}"] = dict(n=int(m.sum()), share_of_zone_points=float(m.sum() / zm.sum()), real_ridge=float(real.sum() / m.sum()), ridge_on_gt=float(fixable.sum() / m.sum()), ridge_also_off=float((real & (dgr > 1.0)).sum() / m.sum()),
                                  no_real_ridge=float((m & ~ok).sum() / m.sum()), median_vmax_without_ridge=float(np.median(vmax[m & ~ok])) if (m & ~ok).any() else None, ridge_worse_than_graph=float(worse.sum() / m.sum()))
    r = res[f"offset_{zname}"]
    print(f"   {zname:14s} n {r['n']:7d} ({100 * r['share_of_zone_points']:4.1f} % of its points): real ridge {100 * r['real_ridge']:4.1f} %  -> ridge point on the GT (<= 1 m) {100 * r['ridge_on_gt']:4.1f} %, ridge also off {100 * r['ridge_also_off']:4.1f} %;"
          f" no real ridge {100 * r['no_real_ridge']:4.1f} % (median profile max {r['median_vmax_without_ridge']})")
print("3. confidence of the nearest node for the offset points vs the good points (dg <= 0.5): median e",
      float(np.median(en[off])), "vs", float(np.median(en[dg <= 0.5])))
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "offset_origin.json"), "w"), indent=1)

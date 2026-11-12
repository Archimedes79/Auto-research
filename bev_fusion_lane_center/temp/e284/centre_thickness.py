"""scratch (answer to Mirko's picture of 2026-11-08, the 1 m lane-centre map of the E279 head): how thick are the predicted centre bands and how filled are the junctions? CPU, every 10th VAL frame (87), the map probability
>= 0.5 (the colouring of the pictures). Per zone (within 8 m of the connector points that the lane GT marks as inside an intersection, else away): predicted cells per metre of GT lane (= the mean band width in metres),
the share of predicted cells farther than 0.75 m / 1.5 m from any GT lane point, and the share of the zone's cells that are predicted (fill).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/centre_thickness.py [--run e279_w_s0]"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qpredict as QP  # noqa: E402

run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "e279_w_s0"
fn = QP.make_pred_fn(run, "exact", "cpu", keep_maps=True)
frames = QE.val_frames()[::10]
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
inbox = (np.abs(cells) <= 48.0).all(1)
acc = {z: dict(cells=0.0, gt_len=0.0, far075=0.0, far15=0.0, zone_cells=0.0, fill=0.0, ideal=0.0) for z in ("near a junction", "away")}
for k, f in enumerate(frames):
    pred = fn(f.token)
    p = np.asarray(pred["maps"]["centre1"][0], float).reshape(-1)
    ref = f.ref()
    nt = ref.nt
    pts = nt["pts"]
    gt_tree = cKDTree(pts) if len(pts) else None
    sp = pts[nt["supp"]]
    stree = cKDTree(sp) if len(sp) else None
    zc = np.where(stree.query(cells)[0] <= 8.0, 0, 1) if stree is not None else np.ones(len(cells), int)
    zp = np.where(stree.query(pts)[0] <= 8.0, 0, 1) if stree is not None else np.ones(len(pts), int)
    on = (p >= 0.5) & inbox
    d_gt = gt_tree.query(cells)[0] if gt_tree is not None else np.full(len(cells), 1e9)
    inb_pts = (np.abs(pts) <= 48.0).all(1)
    for zi, z in enumerate(("near a junction", "away")):
        a = acc[z]
        m = (zc == zi) & inbox
        a["cells"] += float((on & (zc == zi)).sum())
        a["far075"] += float((on & (zc == zi) & (d_gt > 0.75)).sum())
        a["far15"] += float((on & (zc == zi) & (d_gt > 1.5)).sum())
        a["gt_len"] += 0.5 * float(((zp == zi) & inb_pts).sum())
        a["zone_cells"] += float(m.sum())
        a["fill"] += float((on & (zc == zi)).sum())
        a["ideal"] += float(((d_gt <= 1.1774) & (zc == zi) & inbox).sum())          # a perfect prediction of the Gaussian target (sigma 1 m, heat >= 0.5 at 1.18 m)
    f._lane = f._ref = None
    if (k + 1) % 20 == 0:
        print(k + 1, "frames", flush=True)
out = {z: dict(frames=len(frames), width_m=a["cells"] / max(a["gt_len"], 1e-9), far_075=a["far075"] / max(a["cells"], 1), far_15=a["far15"] / max(a["cells"], 1), fill=a["fill"] / max(a["zone_cells"], 1),
               ideal_fill=a["ideal"] / max(a["zone_cells"], 1), ideal_width_m=a["ideal"] / max(a["gt_len"], 1e-9), cells=a["cells"], gt_len_m=a["gt_len"]) for z, a in acc.items()}
json.dump(dict(run=run, result=out), open(os.path.join(ROOT, "temp", "e284", f"centre_thickness_{run}.json"), "w"), indent=1)
for z, v in out.items():
    print(f"{z:16s} band width {v['width_m']:.2f} m   cells farther than 0.75 m from a GT point {100 * v['far_075']:.1f} %   farther than 1.5 m {100 * v['far_15']:.1f} %   fill of the zone {100 * v['fill']:.1f} %   (a perfect prediction of the target: width {v['ideal_width_m']:.2f} m, fill {100 * v['ideal_fill']:.1f} %)")

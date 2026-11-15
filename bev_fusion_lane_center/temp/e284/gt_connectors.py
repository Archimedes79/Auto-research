"""scratch (Mirko's remark of 2026-11-08): what do the GT connectors (the lane centres through the junctions) look like? Over the 873 VAL frames: the connector segments inside an intersection of the lane GT l4: how many per
frame, their length, how much they turn (the heading change between the start and the end of the segment: straight < 60 deg, turn 60 - 120 deg, U-turn-like > 120 deg), and how much of each overlaps another connector
(a point of the segment within 1.0 m of a point of ANOTHER piece).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/gt_connectors.py"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

frames = QE.val_frames()
per_frame, seg_rows = [], []
for f in frames:
    g = f.lane()
    segs = []
    for pi, pc in enumerate(g["pieces"]):
        q = np.asarray(pc["pts"], float)
        for s in pc["segments"]:
            if s["kind"] == "connector" and s["in_intersection"] and s["i1"] - s["i0"] >= 4:
                seg = q[s["i0"]: s["i1"] + 1]
                t0, t1 = seg[min(3, len(seg) - 1)] - seg[0], seg[-1] - seg[max(len(seg) - 4, 0)]
                a = np.degrees(np.arccos(np.clip(t0 @ t1 / (np.linalg.norm(t0) * np.linalg.norm(t1) + 1e-9), -1, 1)))
                L = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
                segs.append((pi, seg, L, float(a)))
    if not segs:
        continue
    trees = [cKDTree(s[1]) for s in segs]
    for i, (pi, seg, L, a) in enumerate(segs):
        ov = np.zeros(len(seg), bool)
        for j, (pj, sj, Lj, aj) in enumerate(segs):
            if pj != pi:
                ov |= trees[j].query(seg)[0] <= 1.0
        seg_rows.append((L, a, float(ov.mean())))
    per_frame.append(len(segs))
    f._lane = f._ref = None
R = np.array(seg_rows)
L, A, OV = R[:, 0], R[:, 1], R[:, 2]
tot = L.sum()
res = dict(frames_with_connectors=len(per_frame), segments=len(R), per_frame_mean=float(np.mean(per_frame)), per_frame_p90=float(np.percentile(per_frame, 90)), length_median=float(np.median(L)),
           share_len_straight=float(L[A < 60].sum() / tot), share_len_turn=float(L[(A >= 60) & (A <= 120)].sum() / tot), share_len_uturn=float(L[A > 120].sum() / tot),
           share_segments_uturn=float((A > 120).mean()), share_segments_overlapping_half=float((OV >= 0.5).mean()), mean_overlap=float(OV.mean()),
           share_uturn_overlapping=float(((A > 120) & (OV >= 0.5)).sum() / max((A > 120).sum(), 1)))
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "gt_connectors.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

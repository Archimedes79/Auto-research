"""Loop 18 / WP4 (one-off): does the recall weight close the far-range recall gap to the control? Recall@1 m of the ground-truth points (every 0.5 m, |x|,|y| < 48) per range bin for j1, j1eq, j1rc, j1rc6, f1 + pruning, rc1 and the control E94 (fork pruning 0.4 for the recall / control variants, 0.3 for the champion and j1 as in the final table; cached raw builder outputs).
Usage (gcar): p18_range_recall.py   -> prints the table and writes runs/e18/range_recall.json"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
H = os.path.expanduser("~/exp_gcar_b123/p18")
ctrl = pickle.load(open(f"{R}/runs/e94_a/graphs_val_clean.pkl", "rb"))["preds"]
SYS = [("j1 (0.4)", "e123_D7_r3_j1", 192, 96, "raw_val873_e123_D7_r3_j1_192.pkl", 0.4), ("j1eq", "e18_D7_j1eq", 192, 96, "raw_val873_e18_D7_j1eq_192.pkl", 0.4),
       ("j1rc", "e18_D7_j1rc", 192, 96, "raw_val873_e18_D7_j1rc_192.pkl", 0.4), ("j1rc6", "e18_D7_j1rc6", 192, 96, "raw_val873_e18_D7_j1rc6_192.pkl", 0.4),
       ("f1 (0.4)", "e123_D7_r3_f1", 288, 144, "raw_val873.pkl", 0.4), ("f1eq", "e18_D7_f1eq", 288, 144, "raw_val873_e18_D7_f1eq_288.pkl", 0.4),
       ("rc1", "e123_D7_r3_rc1", 288, 144, "raw_val873_e123_D7_r3_rc1_288.pkl", 0.4), ("rc2", "e18_D7_rc2", 288, 144, "raw_val873_e18_D7_rc2_288.pkl", 0.4)]
BINS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 70))
found = {}
P_all = None
gts = None
rng_all = []
for nm, name, M, K0, cache, fp in SYS:
    net, cfg = TM.load_builder(M, K0, name)
    raw, gts = pickle.load(open(f"{H}/{cache}", "rb"))
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=fp, **TM.ASM) for r in raw]
    f_list = []
    rg_list = []
    for g, gt in zip(gs, gts):
        P, D, I = GE.dense_points(gt, 0.5)
        if len(P) == 0:
            continue
        Q, _, _ = GE.dense_points(g, 0.5)
        f_list.append(cKDTree(Q).query(P)[0] <= 1.0 if len(Q) else np.zeros(len(P), bool))
        rg_list.append(np.hypot(P[:, 0], P[:, 1]))
    found[nm] = np.concatenate(f_list)
    rng_all = np.concatenate(rg_list)
    print("done", nm, flush=True)
f_list = []
for g, gt in zip(ctrl, gts):
    P, D, I = GE.dense_points(gt, 0.5)
    if len(P) == 0:
        continue
    Q, _, _ = GE.dense_points(g, 0.5)
    f_list.append(cKDTree(Q).query(P)[0] <= 1.0 if len(Q) else np.zeros(len(P), bool))
found["control E94"] = np.concatenate(f_list)
out = {}
print(f"{len(rng_all)} ground-truth points; recall@1 m per range bin")
print("system".ljust(14) + " | all   | " + " | ".join(f"{lo}-{hi} m" for lo, hi in BINS))
for nm, f in found.items():
    row = [f.mean()] + [f[(rng_all >= lo) & (rng_all < hi)].mean() for lo, hi in BINS]
    out[nm] = [float(x) for x in row]
    print(nm.ljust(14) + " | " + " | ".join(f"{x:.3f}" for x in row))
json.dump(dict(bins=BINS, recall=out, n=int(len(rng_all))), open(f"{R}/runs/e18/range_recall.json", "w"), indent=1)

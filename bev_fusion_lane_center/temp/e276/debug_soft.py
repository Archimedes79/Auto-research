import os, sys
import numpy as np, torch
HERE = "/mnt/c/Auto-research/bev_fusion_lane_center/training/grid"
sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../../evaluation")
import qnodes as QN, qgraph_readout as RO
R = 1.5
src = open(HERE + "/test_soft_nms.py").read()
ns = {"__file__": HERE + "/test_soft_nms.py"}
exec(src.split("# 1. the hard limit")[0], ns)
grid_frame, hard_keep, soft_keep, margin = ns["grid_frame"], ns["hard_keep"], ns["soft_keep"], ns["margin"]
rng = np.random.default_rng(0)
for n in range(300):
    pos, e, act = grid_frame(rng)
    if margin(pos, act) < 0.02:
        continue
    h = hard_keep(pos, e)
    s, _ = soft_keep(pos, e)
    if (h != s).any():
        s20, _ = soft_keep(pos, e, iters=20)
        idx = np.argwhere(h != s)
        print("frame", n, "mismatches", len(idx), "with 20 passes:", int((h != s20).sum()))
        for i, j in idx[:3]:
            d = [(a, b, round(float(np.linalg.norm(pos[:, i, j] - pos[:, a, b])), 3), round(float(e[a, b]), 3)) for a in range(max(0, i-1), min(12, i+2)) for b in range(max(0, j-1), min(12, j+2)) if (a, b) != (i, j) and act[a, b]]
            print("  cell", (i, j), "e", round(float(e[i, j]), 3), "hard keep", bool(h[i, j]), "soft keep", bool(s[i, j]), "neighbours (cell, d, e):", d)

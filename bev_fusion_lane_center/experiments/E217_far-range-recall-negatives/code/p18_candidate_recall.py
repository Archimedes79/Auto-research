"""Loop 18 / WP4 test 3b (one-off): do the builder's candidates contain the missing far lanes? Recall@1m of the ground-truth points by range for the path segments (node - parent, densified every 0.5 m) of
(a) all valid nodes, (b) nodes with path validity >= 0.2, (c) >= 0.7 (the admitted ones). Usage (gcar): p18_candidate_recall.py"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
bins = ((0, 20), (20, 40), (40, 70))
acc = {k: {b: [0, 0] for b in bins} for k in ("all valid", "pi>=0.2", "pi>=0.7")}
for r, gt in zip(raw, gts):
    P, _, _ = GE.dense_points(gt, 0.5)
    if len(P) == 0:
        continue
    pos = r["pos"] if "pos_rev" not in r else r["pos_rev"]
    par = r["parent"]
    rg = np.hypot(P[:, 0], P[:, 1])
    for name, thr in (("all valid", 0.0), ("pi>=0.2", 0.2), ("pi>=0.7", 0.7)):
        sel = np.where((r["valid"] > 0) & (r["pi"] >= thr) & (par >= 0))[0]
        if len(sel) == 0:
            continue
        pts = []
        for i in sel:
            a, b = pos[par[i]], pos[i]
            n = max(int(np.linalg.norm(b - a) / 0.5), 1)
            t = np.linspace(0, 1, n + 1)[:, None]
            pts.append(a + t * (b - a))
        Q = np.concatenate(pts)
        f = cKDTree(Q).query(P)[0] <= 1.0
        for lo, hi in bins:
            m = (rg >= lo) & (rg < hi)
            acc[name][(lo, hi)][0] += int(f[m].sum())
            acc[name][(lo, hi)][1] += int(m.sum())
for name, d in acc.items():
    print(f"{name:10s} " + " | ".join(f"{lo}-{hi} m: recall {d[(lo, hi)][0] / max(d[(lo, hi)][1], 1):.3f}" for lo, hi in bins))

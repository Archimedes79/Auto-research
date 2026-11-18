"""Loop 18 / WP4 test 2 (one-off): where are the ground-truth lane points the champion misses and the control finds (R@1.0 0.845 vs 0.894, lanes found 0.695 vs 0.80)?
Per ground-truth point (every 0.5 m, |x|,|y| < 48): found = a predicted point within 1 m. Categories: range bin, multi-lane (another ground-truth lane within 5 m laterally and cos > 0.9),
near a ground-truth junction (< 8 m), short lane piece (ground-truth polyline shorter than 20 m inside the window).
Prints recall by category for both systems and the number of points only the control finds / only the champion finds.
Usage (gcar): p18_missing_lanes.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

RAWP = sys.argv[1] if len(sys.argv) > 1 else "~/exp_gcar_b123/p18/raw_val873.pkl"
FP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
raw, gts = pickle.load(open(os.path.expanduser(RAWP), "rb"))
net, cfg = TM.load_builder(288, 144, sys.argv[3] if len(sys.argv) > 3 else "e123_D7_r3_f1")
R = BP.ROOT
ctrl = pickle.load(open(f"{R}/runs/e94_a/graphs_val_clean.pkl", "rb"))["preds"]
champ = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=FP, **TM.ASM) for r in raw]
rows = []
for gc, gk, gt in zip(champ, ctrl, gts):
    P, D, I = GE.dense_points(gt, 0.5)
    if len(P) == 0:
        continue
    fc = np.zeros(len(P), bool)
    fk = np.zeros(len(P), bool)
    for g, f in ((gc, fc), (gk, fk)):
        Q, _, _ = GE.dense_points(g, 0.5)
        if len(Q):
            f[:] = cKDTree(Q).query(P)[0] <= 1.0
    tree = cKDTree(P)
    multi = np.zeros(len(P), bool)
    for i, nb in enumerate(tree.query_ball_point(P, 5.0)):
        nb = np.array(nb)
        other = nb[I[nb] != I[i]]
        if len(other):
            multi[i] = bool(((D[other] @ D[i]) > 0.9).any())
    sp, mg = GE.junctions(gt)
    jn = np.concatenate([sp, mg]) if len(sp) + len(mg) else np.zeros((0, 2))
    nearj = (cKDTree(jn).query(P)[0] < 8.0) if len(jn) else np.zeros(len(P), bool)
    plen = np.array([np.sum(I == k) * 0.5 for k in range(I.max() + 1)])
    short = plen[I] < 20.0
    rows.append(np.stack([fc, fk, np.hypot(P[:, 0], P[:, 1]), multi, nearj, short], 1))
A = np.concatenate(rows).astype(float)
fc, fk, rg, multi, nearj, short = A.T
N = len(A)
print(f"{N} ground-truth points; recall@1m champion (pruned) {fc.mean():.3f} control {fk.mean():.3f}; only control {int(((fk == 1) & (fc == 0)).sum())} ({((fk == 1) & (fc == 0)).mean():.3f}) only champion {int(((fc == 1) & (fk == 0)).sum())} ({((fc == 1) & (fk == 0)).mean():.3f})")
def line(name, m):
    m = m.astype(bool)
    oc = ((fk == 1) & (fc == 0) & m).sum()
    ok = ((fc == 1) & (fk == 0) & m).sum()
    print(f"{name:26s} n {m.sum():7d} ({m.mean():.2f}) | recall champion {fc[m].mean():.3f} control {fk[m].mean():.3f} | only-control {oc / N:.4f} only-champion {ok / N:.4f} of all points  (net gap share {(oc - ok) / N:+.4f})")
for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 70)):
    line(f"range {lo}-{hi} m", (rg >= lo) & (rg < hi))
line("multi-lane (parallel)", multi == 1)
line("single lane", multi == 0)
line("near junction (<8 m)", nearj == 1)
line("away from junctions", nearj == 0)
line("short GT piece (<20 m)", short == 1)
line("long GT piece", short == 0)

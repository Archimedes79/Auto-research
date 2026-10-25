"""Self-test of sparse_metrics: identical to scripts/p16_common.py, perfect graph scores 1, DP sparsity of the ground truth.
Run: scripts/_py14.sh ../evaluation/sparse_lanes/selftest.py"""
import pickle

import numpy as np

import graph_eval as GE
import p16_common as OLD
import sparse_metrics as SM

HOME = __import__("os").path.expanduser("~")
gts = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
print("gt graphs:", len(gts), "| keys of one graph:", sorted(gts[0].keys()) if isinstance(gts[0], dict) else type(gts[0]))
gts = [dict(pts=[np.asarray(p, np.float64) for p in g["pts"]], edges=list(g["edges"]), nodes=g.get("nodes")) for g in gts[:60]]
rng = np.random.RandomState(0)
pred = [dict(pts=[p + rng.normal(0, 0.4, p.shape) for p in g["pts"]], edges=list(g["edges"]), nodes=None) for g in gts]
for name, P in (("perfect", gts), ("noisy 0.4 m", pred)):
    a, b = OLD.card(P, gts), SM.card(P, gts)
    print(f"{name:12s} identical to p16_common: {a == b} | F1@1.0 {b['F1@1.0']:.3f}  J {b['J']:.3f}  Sp {b['Sp']:.3f}  S {b['S']:.3f}  vert100 {b['vert100']:.1f}")
    assert a == b
print("vert100_dp on the ground-truth graphs (project oracle 14.3 on 100 TEST frames):", round(SM.vert100_dp(gts, 0.3), 2), "| unique end points:", round(SM.vert100_dp(gts, 0.3, unique_ends=True), 2))
pv = [dict(pts=[np.array([[0.0, 0], [10, 0], [20, 0]])], edges=[], nodes=None)]
print("straight 20 m line, dp vertices per 100 m:", round(SM.vert100_dp(pv, 0.3), 2), "(2 end points -> 10.0)")
print("sparse_lanes selftest passed")

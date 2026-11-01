import sys, os, numpy as np
sys.path.insert(0, os.environ["BFLC_ROOT"] + "/evaluation/sparse_lanes")
import topo_metrics as TM
from scipy.spatial import cKDTree
import graph_eval as GE

def old(pred, gt, tau=1.0):
    P, _, _ = GE.dense_points(pred); G, _, _ = GE.dense_points(gt)
    out = np.zeros((len(TM.BINS), 4))
    if not len(G): return out
    dg = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), np.inf)
    dp = cKDTree(G).query(P)[0] if len(P) else np.zeros(0)
    rg, rp = np.linalg.norm(G, axis=1), np.linalg.norm(P, axis=1) if len(P) else np.zeros(0)
    for b, (_, lo, hi) in enumerate(TM.BINS):
        kg, kp = (rg >= lo) & (rg < hi), (rp >= lo) & (rp < hi)
        out[b] = (np.sum(dg[kg] <= tau), kg.sum(), np.sum(dp[kp] <= tau), kp.sum())
    return out

rng = np.random.default_rng(0)
def rg_(n):
    return dict(pts=[np.cumsum(rng.normal(0, 3, (rng.integers(2, 30), 2)), 0) + rng.uniform(-50, 50, 2) for _ in range(n)], edges=[])
bad = 0
for i in range(400):
    p, g = rg_(rng.integers(0, 6)), rg_(rng.integers(0, 6))
    a, b = old(p, g), TM.range_counts(p, g)
    if not np.array_equal(a, b[:3]): bad += 1
    if len(GE.dense_points(g)[0]) and not np.array_equal(b[3] + b[4], b[2]): bad += 1000
print("mismatches", bad)

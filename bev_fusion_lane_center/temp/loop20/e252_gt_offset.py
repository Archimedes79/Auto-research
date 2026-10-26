"""E252 (CPU): how large is the map offset of the lane ground truth, and does it cap the metric by city?

    wsl bash tools/env/_py14.sh temp/loop20/e252_gt_offset.py [registry names]     (default: champion e24_j1eq_s0 e24_j1rc_s0)

Cached raw outputs of the registry models on VAL 873, assembled with the registered settings (evaluate.graphs_of). Two readings per model:
 1. the metric card pooled per city (frames, scenes, Q, F1@0.5 / 1.0 / 2.0, lanes found, R@1_far, vert100) with a scene bootstrap interval of Q (B = 500);
 2. the signed lateral offset of the predicted dense points (0.5 m) that lie within 1.5 m of the ground truth: s = cross(direction of the nearest GT point, P - G),
    positive = left of the lane direction. Per scene the median signed offset (a map registration or localisation error that is common to a scene shows up here as
    a nonzero median; a model that has learned the GT bias hides it, so this is a lower bound), the median |s|, the share of points with |s| > 0.5 m.
Output: a table per model and city, written to experiments/E252_*/gt_offset_<name>.txt by the caller (stdout here). Throw-away script, 1 thread.
"""
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import evaluate as EV  # noqa: E402
import bootstrap as SB  # noqa: E402
import graph_eval as GE  # noqa: E402

LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]
names = sys.argv[1:] or ["champion", "e24_j1eq_s0", "e24_j1rc_s0"]
reg = EV.registry()
import json  # noqa: E402

sp = json.load(open(f"{EV.R}/data/splits.json"))
rows = np.array(sp["val"])
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
loc, scene = pz["loc"][rows], pz["scene"][rows]
print(f"VAL frames {len(rows)}; per city: " + ", ".join(f"{LOCS[c]} {int((loc == c).sum())} frames / {len(np.unique(scene[loc == c]))} scenes" for c in np.unique(loc)))
rng = np.random.RandomState(0)


def offsets(pred, gt):
    P, DP, IP = GE.dense_points(pred)
    G, DG, IG = GE.dense_points(gt)
    if not len(P) or not len(G):
        return np.zeros(0)
    from scipy.spatial import cKDTree
    d, i = cKDTree(G).query(P)
    ok = d <= 1.5
    v = P[ok] - G[i[ok]]
    t = DG[i[ok]]
    return t[:, 0] * v[:, 1] - t[:, 1] * v[:, 0]


_g0, _gt0, _ = EV.graphs_of(names[0], reg[names[0]], "val")
assert max(float(np.abs(offsets(q, q)).max(initial=0.0)) for q in _gt0[:20]) == 0.0, "identity check: the offset of the ground truth against itself must be 0"
assert len(rows) == 873 and sum(int((loc == c).sum()) for c in np.unique(loc)) == 873
print("identity check passed (20 frames); frame counts add to 873")

for name in names:
    g, gts, _ = EV.graphs_of(name, reg[name], "val")
    assert len(g) == len(rows), (len(g), len(rows))
    rec = SB.frame_records(g, gts)
    print(f"\n== {name}")
    print(f"{'city':26} {'frames':>6} {'scenes':>6} {'Q':>7} {'Q 95% scene CI':>16} {'F1@0.5':>7} {'F1@1.0':>7} {'F1@2.0':>7} {'lanes':>6} {'R1far':>6} {'vert100':>7}")
    for c in np.unique(loc):
        idx = np.nonzero(loc == c)[0]
        m = SB.pooled(rec, idx)
        sc = np.unique(scene[idx])
        qs = []
        for _ in range(500):
            pick = rng.choice(sc, len(sc))
            ii = np.concatenate([idx[scene[idx] == s] for s in pick])
            qs.append(SB.pooled(rec, ii)["Q"])
        lo, hi = np.percentile(qs, [2.5, 97.5])
        g_ = lambda k: m.get(k, float("nan"))  # noqa: E731
        print(f"{LOCS[c]:26} {len(idx):6d} {len(sc):6d} {m['Q']:7.4f} [{lo:6.4f},{hi:6.4f}] {g_('F1@0.5'):7.4f} {g_('F1@1.0'):7.4f} {g_('F1@2.0'):7.4f} "
              f"{g_('lanes_found'):6.3f} {g_('R@1_far'):6.3f} {g_('vert100'):7.2f}")
    print(f"{'city':26} {'scenes':>6} {'points':>8} {'median s':>9} {'IQR of scene medians':>22} {'median |s|':>10} {'share |s|>0.5':>13} {'share |s|>1.0':>13} {'scenes with |median s|>0.15':>28}")
    off = [offsets(p, q) for p, q in zip(g, gts)]
    for c in np.unique(loc):
        idx = np.nonzero(loc == c)[0]
        allv = np.concatenate([off[i] for i in idx])
        meds = []
        for s in np.unique(scene[idx]):
            v = np.concatenate([off[i] for i in idx if scene[i] == s])
            if len(v) > 200:
                meds.append(np.median(v))
        meds = np.array(meds)
        q1, q3 = (np.percentile(meds, [25, 75]) if len(meds) else (np.nan, np.nan))
        print(f"{LOCS[c]:26} {len(meds):6d} {len(allv):8d} {np.median(allv):9.3f} [{q1:+.3f},{q3:+.3f}]".ljust(66) +
              f" {np.median(np.abs(allv)):10.3f} {(np.abs(allv) > 0.5).mean():13.3f} {(np.abs(allv) > 1.0).mean():13.3f} {int((np.abs(meds) > 0.15).sum()):>14d} of {len(meds)}")

"""E252 follow-up after the referee (REVISE): controls and a finer statistic. CPU, 1 thread.

    wsl bash tools/env/_py14.sh temp/loop20/e252b_controls.py [registry name, default champion]

1. permutation null: the 22 VAL scenes get the city labels (10 / 12) at random, 2000 draws; the distribution of the Boston - other gap of Q, and the share of draws
   with a gap at least as large as the observed one (the Q gap of the real labels is the primary test, champion; the other models are not independent of it);
2. offset signature: the city gap of F1@0.5 and F1@1.0 (an offset of the ground truth would shrink the gap with the tolerance);
3. synthetic shift control: all ground-truth polylines shifted by s = 0.3 / 0.5 / 1.0 m in x (lateral for a car driving along y), re-scored on all frames: the
   loss of Q and F1@1.0 per metre, to read a Q gap in metres;
4. per ground-truth polyline: the median signed lateral offset of the predicted points matched to it (at least 20 points), distribution of |median| per city, share
   above 0.3 m, and the share of predicted points with no ground-truth point within 1.5 m (the gate of E252 hid them).
"""
import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import evaluate as EV  # noqa: E402
import bootstrap as SB  # noqa: E402
import graph_eval as GE  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import json  # noqa: E402

LOCS = ["boston-seaport", "singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown"]
name = sys.argv[1] if len(sys.argv) > 1 else "champion"
reg = EV.registry()
sp = json.load(open(f"{EV.R}/data/splits.json"))
rows = np.array(sp["val"])
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
loc, scene = pz["loc"][rows], pz["scene"][rows]
g, gts, _ = EV.graphs_of(name, reg[name], "val")
rec = SB.frame_records(g, gts)
B = loc == 0  # boston-seaport
sc_ids = np.unique(scene)
sc_loc = {s: loc[scene == s][0] for s in sc_ids}
print(f"== {name}: Boston {int(B.sum())} frames / {sum(1 for s in sc_ids if sc_loc[s] == 0)} scenes, other {int((~B).sum())} frames / {sum(1 for s in sc_ids if sc_loc[s] != 0)} scenes")


def gap(isB):
    mb, ms = SB.pooled(rec, np.nonzero(isB)[0]), SB.pooled(rec, np.nonzero(~isB)[0])
    return {k: mb[k] - ms[k] for k in ("Q", "F1@0.5", "F1@1.0", "lanes_found") if k in mb}


obs = gap(B)
print("1. observed gap Boston - Singapore:", {k: round(v, 4) for k, v in obs.items()})
rng = np.random.RandomState(0)
nB = sum(1 for s in sc_ids if sc_loc[s] == 0)
null = []
for _ in range(2000):
    pickB = set(rng.choice(sc_ids, nB, replace=False).tolist())
    null.append(gap(np.array([s in pickB for s in scene]))["Q"])
null = np.array(null)
print(f"   permutation null of the Q gap (2000 label draws over scenes): mean {null.mean():+.4f} SD {null.std():.4f}; share with |gap| >= |observed| {(np.abs(null) >= abs(obs['Q'])).mean():.3f}; "
      f"share with gap >= observed {(null >= obs['Q']).mean():.3f}")
print(f"2. offset signature: gap F1@0.5 {obs['F1@0.5']:+.4f}, F1@1.0 {obs['F1@1.0']:+.4f} (ratio {obs['F1@1.0'] / obs['F1@0.5']:.2f}; an offset effect would shrink with the tolerance AND make Boston lower)")

print("3. synthetic shift of the ground truth (all frames):")
for s in (0.0, 0.3, 0.5, 1.0):
    gs = []
    for q in gts:
        d = dict(q)
        d["pts"] = [np.asarray(p, np.float64) + np.array([s, 0.0]) for p in q["pts"]]
        gs.append(d)
    r2 = SB.frame_records(g, gs)
    m = SB.pooled(r2)
    mb, ms = SB.pooled(r2, np.nonzero(B)[0]), SB.pooled(r2, np.nonzero(~B)[0])
    print(f"   shift {s:.1f} m: Q {m['Q']:.4f} (Boston {mb['Q']:.4f}, Singapore {ms['Q']:.4f}), F1@1.0 {m['F1@1.0']:.4f}")

print("4. per ground-truth polyline (predicted points matched within 1.5 m, at least 20 points):")
stat = {0: [], 1: []}
nomatch = {0: [0, 0], 1: [0, 0]}
for i, (p, q) in enumerate(zip(g, gts)):
    c = 0 if B[i] else 1
    P, DP, IP = GE.dense_points(p)
    G, DG, IG = GE.dense_points(q)
    if not len(P) or not len(G):
        continue
    d, j = cKDTree(G).query(P)
    nomatch[c][0] += int((d > 1.5).sum())
    nomatch[c][1] += len(P)
    ok = d <= 1.5
    v = P[ok] - G[j[ok]]
    t = DG[j[ok]]
    s_ = t[:, 0] * v[:, 1] - t[:, 1] * v[:, 0]
    lane = IG[j[ok]]
    for ln in np.unique(lane):
        sel = lane == ln
        if sel.sum() >= 20:
            stat[c].append(np.median(s_[sel]))
for c, lab in ((0, "Boston"), (1, "Singapore")):
    a = np.abs(np.array(stat[c]))
    print(f"   {lab:10} polylines {len(a):6d}: median |polyline median offset| {np.median(a):.3f} m, share > 0.3 m {(a > 0.3).mean():.3f}, share > 0.5 m {(a > 0.5).mean():.3f}; "
          f"predicted points with no GT within 1.5 m: {nomatch[c][0] / max(nomatch[c][1], 1):.3f}")

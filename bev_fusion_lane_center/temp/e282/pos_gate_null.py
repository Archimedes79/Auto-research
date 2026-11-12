"""scratch: would the existing phase-3 position gate (node recall within 2 m of the SPARSE GT nodes >= 0.5, precision within 2 m >= 0.75) pass a zone-correct but freely placed row?
rows: a random cell of each free zone (one node per zone, the GT cell for anchors) = a PERFECT zone-matching head with random choice inside the zone"""
import os, sys
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qtargets as QT
rng = np.random.RandomState(1)
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
c = Counter()
for i in range(0, len(ds), 12):
    g = ds.lg.load(ds.tokens[int(ds.bd.rows[ds.ks[i]])])
    nd = QT.node_table(g); ns = QT.node_table(g, sparse=(0.3, 7.5))
    gd, gs = nd["gid"], ns["gid"]; pieces = nd["piece"]; off = nd["off"]
    pos = {int(a): nd["pos"][m] for m, a in enumerate(gd)}
    anchors = set()
    for p in np.unique(pieces):
        ids = np.where(pieces == p)[0]
        anchors.add(int(gd[ids[0]])); anchors.add(int(gd[ids[-1]]))
    anchors |= set(int(x) for x in gd[nd["ev"]])
    zones = {}
    for p in np.unique(pieces):
        ids = [int(x) for x in gd[pieces == p]]
        sp = sorted(int(x) for x in gs if off[p] <= x < off[p + 1])
        for gid in ids:
            dd = min((abs(gid - s), s) for s in sp); s = dd[1]
            if gid == s or s not in anchors:
                zones.setdefault(s, []).append(gid)
    deliv = []
    for s, ids in zones.items():
        deliv.append(s if s in anchors else int(rng.choice(ids)))
    P = np.array([pos[v] for v in deliv]); G = ns["pos"]
    tg, tp = cKDTree(G), cKDTree(P)
    c["n"] += len(P); c["prec2"] += int((tg.query(P)[0] <= 2.0).sum()); c["rec2"] += int((tp.query(G)[0] <= 2.0).sum()); c["g"] += len(G)
print("perfect zone-matching head (one random cell per free zone, anchors exact): precision within 2 m of the sparse GT", c["prec2"] / c["n"], " recall within 2 m", c["rec2"] / c["g"])

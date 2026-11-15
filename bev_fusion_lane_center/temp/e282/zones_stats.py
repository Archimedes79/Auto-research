"""scratch (E282 critic pass, read-only): zone statistics of the card on real frames of the l4 store (VAL, every 6th frame).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh zones_stats.py
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qtargets as QT  # noqa: E402

ds = LD.QData("val_clean", pixels=False, map_mode="exact")
print("VAL frames with lane GT:", len(ds), "missing", ds.missing)
idx = list(range(0, len(ds), 6))
tot = Counter()
sizes_free = []
adj_pairs = 0
vis_free = Counter()
for n_i, i in enumerate(idx):
    k = ds.ks[i]
    row = int(ds.bd.rows[k])
    token = ds.tokens[row]
    g = ds.lg.load(token)
    nd = QT.node_table(g)
    ns = QT.node_table(g, sparse=(0.3, 7.5))
    gd, gs = nd["gid"], ns["gid"]
    assert set(gs.tolist()) <= set(gd.tolist()), "sparse not subset of dense"
    tot["frames"] += 1
    tot["dense"] += len(gd)
    tot["sparse"] += len(gs)
    pieces = nd["piece"]
    anchors = set()
    for p in np.unique(pieces):
        ids = np.where(pieces == p)[0]
        anchors.add(int(gd[ids[0]]))
        anchors.add(int(gd[ids[-1]]))
    anchors |= set(int(x) for x in gd[nd["ev"]])
    sset = set(int(x) for x in gs)
    assert anchors <= sset, "anchors not all in the sparse set"
    tot["anchor"] += len(anchors)
    tot["free"] += len(sset) - len(anchors)
    off = nd["off"]
    zone_of = {}
    for p in np.unique(pieces):
        ids = [int(x) for x in gd[pieces == p]]
        sp = sorted(int(x) for x in gs if off[p] <= x < off[p + 1])
        for gid in ids:
            d = [(abs(gid - s), s) for s in sp]
            dmin = min(d)[0]
            s = min(x for dd, x in d if dd == dmin)
            if gid == s:
                zone_of[gid] = s
            elif s in anchors:
                zone_of[gid] = None
            else:
                zone_of[gid] = s
    cnt = Counter(v for v in zone_of.values() if v is not None)
    assert all(s in cnt for s in sset), "a sparse node without a zone"
    tot["nozone"] += sum(1 for v in zone_of.values() if v is None)
    for s in sset:
        if s not in anchors:
            sizes_free.append(cnt[s])
    cells = {tuple(c): m for m, c in enumerate(ns["cell"])}
    for (a, b) in cells:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if (da or db) and (a + da, b + db) in cells:
                    adj_pairs += 1
    L = sum(float(np.linalg.norm(np.diff(np.asarray(p["pts"]), axis=0), axis=1).sum()) for p in g["pieces"])
    for m, s in enumerate(gs):
        key = ("anchor_" if int(s) in anchors else "free_") + str(bool(ns["vis"][m]))
        vis_free[key] += 1
    tot["lane_m"] += L
print(dict(tot))
print("dense per 100 m", 100 * tot["dense"] / tot["lane_m"], "sparse per 100 m", 100 * tot["sparse"] / tot["lane_m"])
print("anchors share of sparse nodes", tot["anchor"] / tot["sparse"], "free", tot["free"] / tot["sparse"])
print("dense nodes in no zone:", tot["nozone"], "share of dense", tot["nozone"] / tot["dense"])
sz = np.array(sizes_free)
print("free zone size (cells): mean", sz.mean(), "median", np.median(sz), "dist", {int(v): int((sz == v).sum()) for v in np.unique(sz)[:12]})
print("share of free zones with exactly 1 cell:", float((sz == 1).mean()), " >=3 cells:", float((sz >= 3).mean()))
print("sparse-sparse 8-neighbour cell pairs per frame (ordered pairs):", adj_pairs / tot["frames"])
print("demanded flag of sparse nodes:", dict(vis_free))

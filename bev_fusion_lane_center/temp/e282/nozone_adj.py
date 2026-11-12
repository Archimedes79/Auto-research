"""scratch: are the no-zone dense nodes exactly the non-anchor index-neighbours of anchors? anchors: events vs first/last; long links ends anchors?"""
import os, sys
from collections import Counter
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qtargets as QT
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
c = Counter()
for i in range(0, len(ds), 12):
    g = ds.lg.load(ds.tokens[int(ds.bd.rows[ds.ks[i]])])
    nd = QT.node_table(g); ns = QT.node_table(g, sparse=(0.3, 7.5))
    gd, gs = nd["gid"], ns["gid"]; pieces = nd["piece"]; off = nd["off"]
    anchors = set()
    for p in np.unique(pieces):
        ids = np.where(pieces == p)[0]
        anchors.add(int(gd[ids[0]])); anchors.add(int(gd[ids[-1]]))
    ev = set(int(x) for x in gd[nd["ev"]])
    anchors |= ev
    c["anchors"] += len(anchors); c["anchors_event"] += len(ev); c["anchors_pieceend_only"] += len(anchors - ev)
    for p in np.unique(pieces):
        ids = [int(x) for x in gd[pieces == p]]
        sp = sorted(int(x) for x in gs if off[p] <= x < off[p + 1])
        for pos_i, gid in enumerate(ids):
            dd = min((abs(gid - s), s) for s in sp); s = dd[1]
            if gid != s and s in anchors:
                c["nozone"] += 1
                # index-neighbour (adjacent dense node in the piece) of an anchor?
                nb = [ids[j] for j in (pos_i - 1, pos_i + 1) if 0 <= j < len(ids)]
                c["nozone_adjacent_to_anchor"] += int(any(x in anchors for x in nb))
    # long links in the SPARSE table: ends anchors?
    ga = set(anchors)
    for a, b, lg in zip(ns["src"], ns["dst"], ns["long"]):
        if lg:
            c["long_links_sparse"] += 1
            c["long_ends_anchor"] += int(int(gs[a]) in ga) + int(int(gs[b]) in ga)
    for a, b, lg in zip(nd["src"], nd["dst"], nd["long"]):
        if lg:
            c["long_links_dense"] += 1
            c["long_dense_ends_anchor"] += int(int(gd[a]) in ga) + int(int(gd[b]) in ga)
    # window events at |x|,|y|=50
print(dict(c))

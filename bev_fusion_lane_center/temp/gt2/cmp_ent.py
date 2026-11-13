import gzip
import os
import pickle
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "tools/viz"):
    sys.path.insert(0, os.path.join(R, sub))
import jgt as J
import qgraph_view as QV

store6 = os.path.join(os.path.dirname(R), "temp", "gt_l6_store", "lane", "9ad1d4ce30")
toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk[::2]
n = shape = 0
dpos, dang = [], []
for t in toks:
    g4 = QV.load_gt(t)
    g6 = pickle.load(gzip.open(os.path.join(store6, t + ".pkl.gz"), "rb"))
    j4, j6 = J.junction_gt(g4), J.junction_gt(g6)
    if j4 is None or j6 is None:
        continue
    if j4["ent"].shape != j6["ent"].shape or not np.allclose(j4["ent"], j6["ent"], atol=0.05):
        n += 1
        if j4["ent"].shape != j6["ent"].shape:
            shape += 1
            continue
        d = j4["ent"] - j6["ent"]
        dpos.append(np.abs(d[:, :2]).max())
        dang.append(np.degrees(np.arccos(np.clip((j4["ent"][:, 2:] * j6["ent"][:, 2:]).sum(1), -1, 1))).max())
print("frames with a different entry set", n, "of which shape differs", shape, "max position diff m (median, max)", (np.median(dpos), max(dpos)) if dpos else None, "max heading diff deg (median, max)", (np.median(dang), max(dang)) if dang else None)

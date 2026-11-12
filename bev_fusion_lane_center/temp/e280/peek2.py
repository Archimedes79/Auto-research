"""E280 scratch: facts the pre-check relies on (cells recoverable from pos? dirv unit? road2 shape? sklearn?)"""
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qtargets as QT  # noqa: E402

d = pickle.load(open(os.path.join(ROOT, "temp/e276/predfull_e273_p_s0.pkl"), "rb"))
print("frames", len(d))
n_nodes, coll, edge, nrm = [], 0, 0, []
for x in d:
    e = x["e"]
    n_nodes.append((e >= 0.2).sum())
    c = QT.cell_of(x["pos"])
    key = c[:, 0] * 50 + c[:, 1]
    coll += len(key) - len(np.unique(key))
    off = (x["pos"] - QT.centre_of(c))
    edge += int((np.abs(off) > 0.999).any(1).sum())
    nrm.append(np.linalg.norm(x["dirv"], axis=1))
nrm = np.concatenate(nrm)
print("nodes per frame (e>=0.2): mean %.1f max %d" % (np.mean(n_nodes), np.max(n_nodes)))
print("cell collisions from pos:", coll, " nodes with |offset| > 0.999:", edge)
print("dirv norm min/max", nrm.min(), nrm.max())
x = d[0]
print("keys", sorted(x.keys()), "flags", x["flags"].shape, x["flags"].dtype, "s", None if x["s"] is None else x["s"].shape)
try:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    print("sklearn", sklearn.__version__)
except Exception as ex:
    print("no sklearn:", ex)
try:
    import lightgbm
    print("lightgbm", lightgbm.__version__)
except Exception as ex:
    print("no lightgbm:", ex)

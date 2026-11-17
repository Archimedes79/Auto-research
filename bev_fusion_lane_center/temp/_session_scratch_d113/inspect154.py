import os
import pickle
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "training", "data"))
import lane_tokcache as LC

c = LC.TokCache(os.path.join(R, "runs/e365_cache/be273_p_1_K512_9ad1d4ce30_v1/val"))
d = pickle.load(open(os.path.join(R, "temp/e276/predfull_e273_p_s1.pkl"), "rb"))
i = 154
f = d[i]
n = len(f["e"])
cd = np.clip(np.floor((f["pos"] + 50) / 2).astype(int), 0, 49)
cd = cd[:, 0] * 50 + cd[:, 1]
cc = c.f["cell"][i].astype(int)
ep = c.f["e_p"][i]
print("row", f["row"], c.rows[i], "dump n", n, "cache K", c.K, "n_e20 map", int(c.f["n_e20"][i]), "n_e50", int(c.f["n_e50"][i]))
nc = min(n, c.K)
sd, sc = set(cd[:nc].tolist()), set(cc[:nc].tolist())
print("only in dump", sorted(sd - sc), "only in cache", sorted(sc - sd))
for cell in sorted(sd - sc):
    k = list(cd).index(cell)
    print(" dump cell", cell, "rank", k, "e %.9f" % f["e"][k])
for cell in sorted(sc - sd):
    k = list(cc).index(cell)
    print(" cache cell", cell, "rank", k, "e_p %.9f" % ep[k], "e logit %.9f" % c.f["e"][i][k])
print("last dump e", f["e"][nc - 3:nc], "last cache e_p", ep[nc - 3:nc], "cache e_p behind", ep[nc:nc + 3])

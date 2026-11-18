import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tokcache as LC
import map_sim as MS

c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
lane_gt, _ = LD.lane_store()
T, K, Rn, cnt = [], [], [], []
for i in range(0, len(c), 9):
    g = lane_gt.load(c.tokens[i])
    mp = MS.make_map(g, "error", token=c.tokens[i])
    T.append(len(mp["pos"]))
    K.append(mp["K"])
    Rn.append(mp["n_roads"])
    cnt.extend(mp["pfeat"][:, 3].tolist())
print("frames", len(T), "map tokens per frame mean %.0f max %d | pieces mean %.0f max %d | roads mean %.1f max %d" % (np.mean(T), max(T), np.mean(K), max(K), np.mean(Rn), max(Rn)))
print("lane count values (pfeat[:, 3] histogram):", np.unique(np.round(cnt, 2), return_counts=True))
mp = MS.make_map(lane_gt.load(c.tokens[100]), "exact", token=c.tokens[100])
print("pfeat example rows:\n", np.round(mp["pfeat"][:4], 2), "\nkeys:", sorted(mp.keys()))

import os, sys, time, json, hashlib
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tokcache as LC
import sd_lib as SD
import jtok
print("HALF", jtok.HALF)
c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
lane_gt, d = LD.lane_store()
print(d, len(c))
t0 = time.time()
n = []
for i in range(0, 873, 40):
    g = lane_gt.load(c.tokens[i])
    t1 = time.time()
    sd = SD.build_sd(dict(pts=[np.asarray(p["pts"], float) for p in g["pieces"]]), [])
    t2 = time.time()
    n.append((len(g["pieces"]), len(sd["roads"]), t1 - t0, t2 - t1))
    t0 = time.time()
print(n)
print(sorted(g.keys()), sorted(g["pieces"][0].keys()))
print(np.mean([x[2] for x in n]), np.mean([x[3] for x in n]))

import os
import sys
import time

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_tok_data as LT
import lane_tokcache as LC
import lane_ptok_data as LP

c = LC.TokCache(LP.cache_dir("train", 0, "e292_a"))
ds = LT.TokData("train", res=2.0, feats=False)
idx = np.linspace(0, len(c) - 1, 40).astype(int)
tk = {k: np.array(c.f[k][int(i)]) for i in idx[:1] for k in ("pos", "dirv", "valid", "e_p")}
ts = []
tl = []
for i in idx:
    tk = {k: np.array(c.f[k][int(i)], float if k != "valid" else bool) for k in ("pos", "dirv", "valid", "e_p")}
    t0 = time.time()
    t = ds.table(int(i))
    tl.append(time.time() - t0)
    t0 = time.time()
    LC.label_frame(tk, t, K=512, allowed=np.zeros((512, 512), bool))
    ts.append(time.time() - t0)
print("label_frame per frame: median %.1f ms, mean %.1f ms, p90 %.1f ms | table read (cached npz) median %.1f ms" % (1000 * np.median(ts), 1000 * np.mean(ts), 1000 * np.percentile(ts, 90), 1000 * np.median(tl)))

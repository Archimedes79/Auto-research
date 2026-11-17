import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_ptok_data as LP, lane_tokcache as LC
import qtargets as QT
ds = LP.PTokData("val", 0, "e292_a")
f = ds.c.f; K = ds.K
def T(fn, n=40):
    fn(0)
    t0 = time.perf_counter()
    for i in range(n): fn(i * 7 + 3)
    return (time.perf_counter() - t0) / n * 1000
print("read token fields ms", T(lambda i: [np.array(f[n][i]) for n in LP.TOKEN_FIELDS]))
print("read pair fields ms", T(lambda i: (np.array(f["ypair"][i]), np.array(f["ypair_long"][i]), np.array(f["ypair_scale"][i]))))
print("read label fields ms", T(lambda i: [np.array(f[n][i]) for n in ("gt_id","gd","gt_flags","gt_aux","gt_type","pos","dirv","e_p","flags_p","s_p","cell")]))
yp, yl_, ys_ = np.array(f["ypair"][5]), np.array(f["ypair_long"][5]), np.array(f["ypair_scale"][5])
print("densify ms", T(lambda i: LP.densify(yp, yl_, ys_, K)))
pos = np.array(f["pos"][5], np.float64)
y = np.zeros((K, K), np.uint8)
print("chord ms", T(lambda i: (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)))
print("full item ms", T(lambda i: ds[i]))
print("collate8 ms", T(lambda i: LP.collate([ds[j] for j in range(8)]), 3) )

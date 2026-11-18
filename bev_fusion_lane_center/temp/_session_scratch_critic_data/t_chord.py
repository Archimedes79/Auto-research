import os, sys, time
import numpy as np
import torch
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_ptok_data as LP, qtargets as QT
ds = LP.PTokData("val", 0, "e292_a")
f = ds.c.f; K = ds.K
def sparse_ylong(ypair, pos, K):
    ok = ypair[:, 0] >= 0
    i, j = ypair[ok, 0].astype(int), ypair[ok, 1].astype(int)
    yl = np.zeros((K, K), bool)
    yl[i, j] = np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT
    return yl
bad = 0; tm_d = tm_s = 0
for n in range(0, 873, 7):
    pos = np.array(f["pos"][n], np.float64); yp = np.array(f["ypair"][n])
    y, _, _ = LP.densify(yp, np.array(f["ypair_long"][n]), np.array(f["ypair_scale"][n]), K)
    t0 = time.perf_counter(); dense = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT); tm_d += time.perf_counter() - t0
    t0 = time.perf_counter(); sp = sparse_ylong(yp, pos, K); tm_s += time.perf_counter() - t0
    bad += int((dense != sp).sum())
print("frames", len(range(0, 873, 7)), "differing entries", bad, "dense ms", tm_d / 125 * 1000, "sparse ms", tm_s / 125 * 1000)
# the pair-diff sign: dense is pos[j]-pos[i] at [i, j]? index [a,b] = pos[None,:,:][b] - pos[:,None,:][a] = pos[b]-pos[a]

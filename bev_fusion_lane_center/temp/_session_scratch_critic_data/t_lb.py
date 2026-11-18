import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_ptok_data as LP, lane_live_tok as LL, lane_tok_data as LT, qtargets as QT
pt = LP.PTokData("val", 0, "e292_a"); K = pt.K
td = LT.TokData("val", res=2.0, feats=False)
idx = [3, 50, 200, 400, 600, 800, 120, 300]
tabs = [td.table(i) for i in idx]
cached = {k: torch.from_numpy(np.stack([np.array(pt.c.f[k][i]) for i in idx])) for k in ("cell", "valid", "pos", "dirv", "e_p", "flags_p", "s_p")}
toks = [td.tokens[i] for i in idx]; rows = [td.rows[i] for i in idx]
LL.label_batch(cached, tabs, toks, rows, K, gate=False)
t0 = time.perf_counter()
for _ in range(3): _, labs, secs = LL.label_batch(cached, tabs, toks, rows, K, gate=False)
tot = (time.perf_counter() - t0) / 3
print("label_batch(8, gate False) ms total", tot * 1000, "per frame", tot / 8 * 1000)
pos = cached["pos"][0].numpy().astype(np.float64)
t0 = time.perf_counter()
for _ in range(24): np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT
print("chord per frame ms", (time.perf_counter() - t0) / 24 * 1000)

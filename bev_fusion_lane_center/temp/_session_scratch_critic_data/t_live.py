import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_live_tok as LL
t0 = time.perf_counter(); ds = LL.LiveData("val", 0, "e292_a"); print("LiveData open s", time.perf_counter() - t0)
for i in range(3): ds[i]
t0 = time.perf_counter()
items = [ds[i * 20] for i in range(16)]
print("item ms (x + table)", (time.perf_counter() - t0) / 16 * 1000)
t0 = time.perf_counter()
for _ in range(3): b = LL.collate(items[:8])
print("collate(8) ms (node_targets)", (time.perf_counter() - t0) / 3 * 1000)

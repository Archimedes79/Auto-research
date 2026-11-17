import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
torch.set_num_threads(2)
for split in ("train", "val"):
    ds, dl = LL.loader(split, 0, bs=8, workers=2)
    t0 = time.time()
    it = iter(dl)
    ts = []
    for k in range(4):
        t1 = time.time()
        b = next(it)
        ts.append(time.time() - t1)
    print(split, len(ds), "batches:", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else type(v).__name__ for k, v in b.items() if k != "targets"}, {k: tuple(v.shape) for k, v in b["targets"].items()})
    print("  seconds per batch of 8 (workers 2):", [round(t, 2) for t in ts], "first at", round(time.time() - t0, 1))
    del it, dl

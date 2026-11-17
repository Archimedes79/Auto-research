import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/data/prep", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, s)))
import lane_map_data as LM
ds = LM.MapData("val", 0, "e292_a", None, "train")
for i in range(3): ds[i]
for mode in ("exact", "error", "train", "S"):
    t0 = time.perf_counter()
    for i in range(0, 600, 20): ds.table(i, mode)
    print(mode, "ms/map", (time.perf_counter() - t0) / 30 * 1000)
t0 = time.perf_counter()
for i in range(0, 600, 20): ds[i]
print("MapData item (train mode, incl PTokData) ms", (time.perf_counter() - t0) / 30 * 1000)

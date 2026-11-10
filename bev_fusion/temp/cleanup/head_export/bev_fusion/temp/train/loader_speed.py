"""One-off: samples per second of the cached training loader (no model) for a few worker counts, and of the model step alone."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "training")
from train import TrainConfig, make_datasets, make_loader  # noqa: E402

cfg = TrainConfig(**json.load(open("temp/train/speed_b4.json")))
train_ds, _ = make_datasets(cfg)
for w in [int(a) for a in sys.argv[1:]] or [4, 8]:
    cfg.workers = w
    it = iter(make_loader(train_ds, cfg, True))
    next(it)
    t0, n = time.time(), 0
    for _ in range(24):
        n += len(next(it)["tokens"])
    print(f"workers {w}: {n / (time.time() - t0):.1f} samples/s", flush=True)
    del it

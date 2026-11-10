"""One-off: where does the loading time of one cached training sample go? (camera frames / radar sweeps / GT npz /
feature store), single process, 24 train samples."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "training")
from train import TrainConfig, make_datasets  # noqa: E402

cfg = TrainConfig(**json.load(open("training/configs/t1_r0.json")))
ds, _ = make_datasets(cfg)
T = {"cam_frames": 0.0, "radar": 0.0, "gt": 0.0, "features": 0.0, "total": 0.0}
idx = np.linspace(0, len(ds) - 1, 24).round().astype(int)
for i in idx:
    tok = ds.samples[i]
    t0 = time.time()
    sample = ds.nusc.get("sample", tok)
    frames, track, cur = ds._cam_frames(tok)
    t1 = time.time()
    ds._radar(sample, cur)
    t2 = time.time()
    ds._gt(tok)
    t3 = time.time()
    for f in frames:
        ds.store.get(f["token"])
    t4 = time.time()
    T["cam_frames"] += t1 - t0
    T["radar"] += t2 - t1
    T["gt"] += t3 - t2
    T["features"] += t4 - t3
    t5 = time.time()
    ds[i]
    T["total"] += time.time() - t5
print({k: f"{1000 * v / len(idx):.0f} ms" for k, v in T.items()}, "per sample")

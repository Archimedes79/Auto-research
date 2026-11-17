import json
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "training", "data"))
import lane_seq_data as LS
import p16_data as PD

for split in ("val", "train"):
    fr = json.load(open(os.path.join(R, f"runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/{split}/frames.json")))
    rows = fr["rows"]
    pose = dict(np.load(f"{PD.D}/poses_{split}.npz"))
    so = LS.stream_order(pose, rows)
    same = [int(r) for r, _ in so] == [int(r) for r in rows]
    prev = pose["prev"]
    consecutive = sum(int(prev[r]) == int(rows[i - 1]) for i, r in enumerate(rows) if i > 0 and int(prev[r]) >= 0)
    haspast = sum(int(prev[r]) in set(rows) for r in rows)
    print(split, "cache order == stream order (scene, time):", same, "| frames with a past", haspast, "| of them whose past is the row right before", consecutive)

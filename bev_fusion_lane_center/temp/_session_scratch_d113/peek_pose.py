import json
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "training", "data"))
import p16_data as PD

print("PD.D", PD.D)
for part in ("train", "val"):
    z = dict(np.load(f"{PD.D}/poses_{part}.npz"))
    print(part, {k: (v.shape, str(v.dtype)) for k, v in z.items()})
    prev = z["prev"]
    print("  rows", len(prev), "scene starts", int((prev < 0).sum()), "ts range", float(z["ts"].min()), float(z["ts"].max()))
    # the step between consecutive keyframes
    ok = prev >= 0
    dt = z["ts"][ok] - z["ts"][prev[ok]]
    print("  dt s: median %.3f min %.3f max %.3f" % (np.median(dt), dt.min(), dt.max()))
    d = np.linalg.norm(z["t"][ok] - z["t"][prev[ok]], axis=1)
    print("  ego step m: median %.2f p95 %.2f max %.2f" % (np.median(d), np.percentile(d, 95), d.max()))
c = json.load(open(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val/frames.json")))
print("cache val frames", len(c["rows"]), c["rows"][:3], c["tokens"][:2])
c = json.load(open(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/train/frames.json")))
print("cache train frames", len(c["rows"]), c["rows"][:3])
z = dict(np.load(f"{PD.D}/poses_train.npz"))
rows = set(c["rows"])
print("train cache rows whose prev row is in the cache:", sum(int(z["prev"][r]) in rows for r in c["rows"]), "of", len(rows))

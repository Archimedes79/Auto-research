"""One-off (t3 weak spot: velocity): is the predicted velocity of matched vehicles better than predicting zero?
Uses the stored scene records (dt 0) of finished runs; true positives at 2 m (the DetMeter rule: evaluation.metrics.match)."""
import math
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from evaluation.metrics import in_front, match  # noqa: E402

for run in sys.argv[1:]:
    rec = pickle.load(open(f"runs/{run}/eval/scene_records.pkl", "rb"))["det"]
    frames = []
    for s, r in rec.items():
        for ds, gs in r.get(0.0, []):
            frames.append(([d for d in ds if d["group"] == "vehicle" and in_front(d)], [g for g in gs if g["group"] == "vehicle" and in_front(g)]))
    m, n_gt = match(frames, 2.0)
    pe, ze, sp, mov = [], [], [], []
    for sc, tp, d, g in m:
        if g is None or not tp:
            continue
        gv = math.hypot(g["vx"], g["vy"])
        if not (math.isfinite(gv) and math.isfinite(d["vx"])):
            continue
        pe.append(math.hypot(d["vx"] - g["vx"], d["vy"] - g["vy"]))
        ze.append(gv)
        sp.append(math.hypot(d["vx"], d["vy"]))
        mov.append(gv > 1.0)
    pe, ze, mov = np.array(pe), np.array(ze), np.array(mov)
    print(f"{run}: {len(pe)} matched vehicles ({int(mov.sum())} moving > 1 m/s): AVE predicted {pe.mean():.2f} m/s | zero-velocity baseline {ze.mean():.2f} m/s | "
          f"moving only: predicted {pe[mov].mean():.2f} vs zero {ze[mov].mean():.2f} | mean |GT v| moving {ze[mov].mean():.2f}, predicted speed {np.mean(sp):.2f}")

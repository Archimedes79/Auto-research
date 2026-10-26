"""One-off: heading error of the matched vehicle detections of runs/<name>/eval/scene_records.pkl (dt 0, 2 m):
full error (period 2 pi), axis error (period pi), share of flips (> 90 deg), and the same against the ego heading prior
(GT yaw near 0 or pi: most vehicles drive along the road)."""
import math
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from evaluation.metrics import in_front, match  # noqa: E402

rec = pickle.load(open(f"runs/{sys.argv[1]}/eval/scene_records.pkl", "rb"))
frames = []
for s, r in rec["det"].items():
    for ds, gs in r[0.0]:
        frames.append(([d for d in ds if d["group"] == "vehicle" and in_front(d)],
                       [g for g in gs if g["group"] == "vehicle" and in_front(g)]))
m, n_gt = match(frames, 2.0)
pairs = [(d, g) for _, ok, d, g in m if ok and d["score"] >= 0.3]
full = np.array([abs(math.remainder(d["yaw"] - g["yaw"], 2 * math.pi)) for d, g in pairs])
axis = np.array([abs(math.remainder(d["yaw"] - g["yaw"], math.pi)) for d, g in pairs])
gy = np.array([g["yaw"] for _, g in pairs])
dy = np.array([d["yaw"] for d, _ in pairs])
print(f"{len(pairs)} matched vehicles (score >= 0.3): full error mean {full.mean():.3f} rad, median {np.median(full):.3f}; "
      f"axis error mean {axis.mean():.3f}, median {np.median(axis):.3f}; flips (> 90 deg) {np.mean(full > math.pi / 2):.2f}")
print("GT yaw |cos| > 0.9 (along x):", np.mean(np.abs(np.cos(gy)) > 0.9).round(2),
      " predicted:", np.mean(np.abs(np.cos(dy)) > 0.9).round(2))
print("GT yaw histogram (deg, 8 bins):", np.histogram(np.degrees(gy), bins=8, range=(-180, 180))[0])
print("pred yaw histogram (deg, 8 bins):", np.histogram(np.degrees(dy), bins=8, range=(-180, 180))[0])
ss = np.sign(np.sin(dy)) == np.sign(np.sin(gy))
cc = np.sign(np.cos(dy)) == np.sign(np.cos(gy))
print(f"sin sign agrees {ss.mean():.2f}, cos sign agrees {cc.mean():.2f}; mean sin(d)sin(g) {np.mean(np.sin(dy)*np.sin(gy)):.2f}, "
      f"mean cos(d)cos(g) {np.mean(np.cos(dy)*np.cos(gy)):.2f}")
mov = np.array([math.hypot(g["vx"], g["vy"]) > 1.0 for _, g in pairs])
print(f"moving GT (> 1 m/s): {mov.mean():.2f} of matches; flips among moving {np.mean(full[mov] > math.pi/2):.2f}, among parked {np.mean(full[~mov] > math.pi/2):.2f}")
vdir = np.array([math.atan2(g["vy"], g["vx"]) for _, g in pairs])
print(f"moving GT: heading vs velocity direction error median {np.median(np.abs(np.remainder(gy[mov]-vdir[mov]+math.pi, 2*math.pi)-math.pi)):.2f} rad")

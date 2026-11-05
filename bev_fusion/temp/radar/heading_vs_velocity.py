"""One-off: is the velocity of a moving object along its heading? (basis of a heading-conditioned speed: v = s * h, the Doppler gives s)
nuScenes mini annotations: velocity by finite differences of the same instance between neighbouring keyframes (2 Hz, so noisy: the angles below are an upper bound
of the true misalignment), heading from the annotation quaternion. Per group: share of moving objects (|v| > 1 m/s) driving forward, angle between velocity and the
heading AXIS, and the tangential speed (|v| sin(angle)) a perfect heading prior would leave unexplained.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/heading_vs_velocity.py"""
import json
import math
import os

import numpy as np

root = os.path.expanduser("~/data/nuscenes/v1.0-mini")
J = lambda n: json.load(open(os.path.join(root, n + ".json")))
sample = {s["token"]: s["timestamp"] * 1e-6 for s in J("sample")}
ann = {a["token"]: a for a in J("sample_annotation")}
inst = {i["token"]: i["category_token"] for i in J("instance")}
cat = {c["token"]: c["name"] for c in J("category")}


def group(name):
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return "two-wheeler"
    if name.startswith("vehicle.") and "construction" not in name and "emergency" not in name:
        return "car/truck/bus/trailer"
    return None


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


rows = {}
for a in ann.values():
    g = group(cat[inst[a["instance_token"]]])
    if g is None:
        continue
    p, n = ann.get(a["prev"]), ann.get(a["next"])
    lo, hi = (p or a), (n or a)
    dt = sample[hi["sample_token"]] - sample[lo["sample_token"]]
    if dt <= 0:
        continue
    v = (np.array(hi["translation"][:2]) - np.array(lo["translation"][:2])) / dt
    sp = float(np.linalg.norm(v))
    if sp < 1.0:
        continue
    d = math.atan2(v[1], v[0]) - yaw_of(a["rotation"])
    d = (d + math.pi) % (2 * math.pi) - math.pi
    ang = min(abs(d), math.pi - abs(d))
    om = ((yaw_of(hi["rotation"]) - yaw_of(lo["rotation"]) + math.pi) % (2 * math.pi) - math.pi) / dt        # yaw rate of the box (rad/s), one-sided at the ends
    turning = abs(math.degrees(om)) >= 8.0 or (sp < 5.0 and abs(math.degrees(om)) >= 5.0)                    # a curve: clearly rotating (deg/s)
    rows.setdefault((g, "turning" if turning else "straight"), []).append((abs(d) < math.pi / 2, ang, sp, sp * math.sin(ang)))

for (g, kind), r in sorted(rows.items()):
    r = np.array(r, dtype=float)
    ang = np.degrees(r[:, 1])
    print(f"{g:22s} {kind:8s} n={len(r):4d}  forward {100 * r[:, 0].mean():5.1f} %  angle to the heading axis: median {np.median(ang):5.1f}  p90 {np.percentile(ang, 90):5.1f}  "
          f"< 10 deg {100 * (ang < 10).mean():5.1f} %  < 20 deg {100 * (ang < 20).mean():5.1f} %   speed median {np.median(r[:, 2]):4.1f} m/s   "
          f"tangential rest: median {np.median(r[:, 3]):.2f}  p90 {np.percentile(r[:, 3], 90):.2f} m/s")

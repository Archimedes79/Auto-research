"""One-off: is nuScenes radar Doppler folded (ambiguous), and what is its interval?

The radial velocity is v_r = (vx, vy) . u with u the unit line of sight in the sensor frame (vx, vy are the radial velocity
vector, not compensated). Prints per ambig_state the count and the v_r quantiles / extremes, and a histogram of |v_r| near
the largest values, to see a hard limit (the unambiguous interval) or folded values.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/doppler_stats.py [N per radar]
"""
import json
import os
import random
import sys

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

root = os.path.expanduser("~/data/nuscenes_full")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
sd = json.load(open(os.path.join(root, "v1.0-trainval", "sample_data.json")))
RadarPointCloud.disable_filters()
random.seed(1)
allp = []
for ch in ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]:
    files = [r["filename"] for r in sd if r["filename"].split("/")[1] == ch]
    for f in random.sample(files, n * 2):
        p = os.path.join(root, f)
        if os.path.exists(p):
            allp.append(RadarPointCloud.from_file(p).points)
P = np.concatenate(allp, 1)
x, y, vx, vy, amb, vxc, vyc = P[0], P[1], P[6], P[7], P[11].astype(int), P[8], P[9]
r = np.hypot(x, y)
u = np.stack([x, y]) / np.maximum(r, 1e-6)
vr = vx * u[0] + vy * u[1]                     # radial velocity, raw (sensor moves with the ego)
vrc = vxc * u[0] + vyc * u[1]                  # ego-compensated radial velocity
perp = np.abs(-vx * u[1] + vy * u[0])          # component of (vx, vy) across the line of sight (0 if it is radial)
print(f"{P.shape[1]} detections; |(vx,vy) across the line of sight| 99 % quantile {np.quantile(perp, 0.99):.3f} m/s")
for a in sorted(set(amb)):
    m = amb == a
    q = np.quantile(vr[m], [0.001, 0.5, 0.999])
    print(f"ambig_state {a}: {m.sum()} detections; raw v_r 0.1/50/99.9 % {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} m/s, "
          f"min {vr[m].min():.1f}, max {vr[m].max():.1f}; compensated min {vrc[m].min():.1f}, max {vrc[m].max():.1f}")
h, e = np.histogram(np.abs(vr), bins=np.arange(0, 80, 5))
print("|raw v_r| histogram (5 m/s bins from 0):", " ".join(str(c) for c in h))

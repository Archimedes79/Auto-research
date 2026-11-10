"""One-off: measure the effective resolution of nuScenes radar detections in the sensor's polar frame.

For each radar, over N random sweeps (no filters): the smallest spacings between two detections in range (at nearly the same
azimuth) and in azimuth (at nearly the same range), as quantiles. Detections of one target are spread over a few bins, so the
low quantiles give the sensor's resolution cell, and the median spacing of near neighbours the typical footprint size.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_resolution.py [N]
"""
import json
import os
import random
import sys

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

root = os.path.expanduser("~/data/nuscenes_full")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
sd = json.load(open(os.path.join(root, "v1.0-trainval", "sample_data.json")))
RadarPointCloud.disable_filters()
random.seed(2)
for ch in ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_BACK_LEFT"]:
    files = [r["filename"] for r in sd if r["filename"].split("/")[1] == ch]
    d_phi_by_range = {"<20 m": [], "20-40 m": [], ">40 m": []}
    d_r = []
    for f in random.sample(files, n * 2):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        P = RadarPointCloud.from_file(p).points
        r = np.hypot(P[0], P[1])
        ph = np.degrees(np.arctan2(P[1], P[0]))
        dr = np.abs(r[:, None] - r[None, :])
        dph = np.abs(ph[:, None] - ph[None, :])
        np.fill_diagonal(dr, np.inf)
        np.fill_diagonal(dph, np.inf)
        same_range = dr < 0.3
        for name, lo, hi in [("<20 m", 0, 20), ("20-40 m", 20, 40), (">40 m", 40, 300)]:
            m = same_range & ((r[:, None] >= lo) & (r[:, None] < hi))
            d_phi_by_range[name].extend(dph[m & (dph > 1e-3)].tolist())
        same_az = dph < 0.2
        d_r.extend(dr[same_az & (dr > 1e-3)].tolist())
    q = lambda a: " / ".join(f"{v:.2f}" for v in np.quantile(a, [0.01, 0.05, 0.25])) if len(a) else "n/a"
    print(f"{ch}: azimuth spacing at the same range (1/5/25 % quantile, deg): " +
          "; ".join(f"{k} {q(v)} (n={len(v)})" for k, v in d_phi_by_range.items()) +
          f" | range spacing at the same azimuth (m): {q(d_r)} (n={len(d_r)})")

"""One-off: statistics of nuScenes radar detections in their own sensor frame, to size a polar radar grid.

Reads N random sweeps per radar channel with the devkit (no filters: all detections), prints per channel: detections per
sweep, range and azimuth quantiles, the x_rms / y_rms / vx_rms / vy_rms value sets, and the share of detections beyond 60 m.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_stats.py [N]
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
random.seed(0)
fields = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp", "is_quality_valid",
          "ambig_state", "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms"]
for ch in ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]:
    files = [r["filename"] for r in sd if r["filename"].startswith(f"sweeps/{ch}/") or r["filename"].startswith(f"samples/{ch}/")]
    files = [f for f in random.sample(files, min(len(files) * 3, n * 3)) if os.path.exists(os.path.join(root, f))][:n]
    pts, counts = [], []
    for f in files:
        p = RadarPointCloud.from_file(os.path.join(root, f)).points      # (18, N) in the sensor frame
        counts.append(p.shape[1])
        pts.append(p)
    P = np.concatenate(pts, 1)
    d = dict(zip(fields, P))
    r = np.hypot(d["x"], d["y"])
    az = np.degrees(np.arctan2(d["y"], d["x"]))
    q = lambda a: " / ".join(f"{v:.1f}" for v in np.quantile(a, [0.01, 0.5, 0.99]))
    print(f"{ch}: {len(files)} sweeps, detections per sweep median {np.median(counts):.0f} (max {max(counts)}); "
          f"range 1/50/99 % {q(r)} m, beyond 60 m {np.mean(r > 60) * 100:.1f} %; azimuth 1/50/99 % {q(az)} deg; "
          f"z unique {np.unique(np.round(d['z'], 2))[:3]}")
    for k in ["x_rms", "y_rms", "vx_rms", "vy_rms", "ambig_state", "invalid_state", "dyn_prop"]:
        u, c = np.unique(d[k].astype(int), return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:6]
        print(f"   {k}: " + ", ".join(f"{v}:{cnt}" for cnt, v in top))

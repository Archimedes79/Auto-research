"""empirical field of view of the five nuScenes radars from the detections of mini (all samples + sweeps)"""
import math
import os
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

root = os.path.expanduser("~/data/nuscenes")
nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
acc = {}
for sd in nusc.sample_data:
    if "RADAR" not in sd["channel"] if "channel" in sd else "RADAR" not in sd["filename"]:
        continue
    ch = sd["filename"].split("/")[1]
    pts = RadarPointCloud.from_file(os.path.join(root, sd["filename"]), ).points
    if pts.shape[1] == 0:
        continue
    ang = np.degrees(np.arctan2(pts[1], pts[0]))
    rng = np.hypot(pts[0], pts[1])
    acc.setdefault(ch, []).append(np.stack([ang, rng]))
for ch, v in sorted(acc.items()):
    a = np.concatenate(v, 1)
    cs = nusc.get("calibrated_sensor", nusc.get("sample_data", next(s["token"] for s in nusc.sample_data if ch in s["filename"]))["calibrated_sensor_token"])
    yaw = math.degrees(Quaternion(cs["rotation"]).yaw_pitch_roll[0])
    print(f"{ch:18s} mount yaw {yaw:+6.1f} deg | n {a.shape[1]:6d} | angle (deg) 0.1 / 99.9 pct {np.percentile(a[0], 0.1):+6.1f} {np.percentile(a[0], 99.9):+6.1f}, "
          f"max |angle| {np.abs(a[0]).max():5.1f} | range (m) median {np.median(a[1]):5.1f}, 99 pct {np.percentile(a[1], 99):5.1f}, 99.9 pct {np.percentile(a[1], 99.9):5.1f}, max {a[1].max():5.1f}")

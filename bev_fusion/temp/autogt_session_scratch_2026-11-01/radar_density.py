"""scratch (D36): where does the radar actually deliver detections? Static coverage of the rig from nuScenes mini (404 keyframes x 5 radars).

Run from C:\\Auto-research in WSL:  wsl bash autogt/run_nice.sh bev_fusion/temp/autogt_session_scratch_2026-11-01/radar_density.py
Per keyframe: the detections of the five radar sensor samples (devkit default filter) in the ego frame; a cell counts as "measured" when a
detection lies within R m of it. p(cell) = the share of keyframes in which it was measured. Prints area shares for thresholds and range bins
and writes radar_density.png (the fan of the GT, the density, the thresholded masks) next to this script.
"""
import os
import sys

import cv2
import matplotlib
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from autogt import view  # noqa: E402
from autogt.nuscenes_gt import RADAR_FOV  # noqa: E402

root = os.path.expanduser("~/data/nuscenes")
nusc = NuScenes(version="v1.0-mini", dataroot=root, verbose=False)
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
NX, NY, RES = 120, 104, 0.5
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(R / RES) + 1, 2 * int(R / RES) + 1))
acc = np.zeros((NX, NY), np.float64)
npts, n = [], 0
for sc in nusc.scene:
    tok = sc["first_sample_token"]
    while tok:
        sample = nusc.get("sample", tok)
        hit = np.zeros((NX, NY), np.uint8)
        cnt = 0
        for ch in RADARS:
            sd = nusc.get("sample_data", sample["data"][ch])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            pts = RadarPointCloud.from_file(os.path.join(root, sd["filename"])).points
            P = Quaternion(cs["rotation"]).rotation_matrix @ pts[:3] + np.array(cs["translation"])[:, None]      # sensor -> ego (x forward, y left)
            ix, iy = np.floor(P[0] / RES).astype(int), np.floor((P[1] + 26.0) / RES).astype(int)
            ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
            hit[ix[ok], iy[ok]] = 1
            cnt += int(ok.sum())
        acc += cv2.dilate(hit, k)
        npts.append(cnt)
        n += 1
        tok = sample["next"]
p = acc / n
xs = (np.arange(NX) + 0.5) * RES
ys = -26 + (np.arange(NY) + 0.5) * RES
gx, gy = np.meshgrid(xs, ys, indexing="ij")
half, rmax = RADAR_FOV
fan = (np.abs(np.arctan2(gy, np.maximum(gx, 1e-6))) <= half) & (np.hypot(gx, gy) <= rmax) | (gx < 0)
fan = (np.hypot(gx, gy) <= rmax) & (np.abs(np.arctan2(gy, gx)) <= half)
print(f"keyframes {n}, detections inside the grid per keyframe: median {int(np.median(npts))}, mean {np.mean(npts):.0f}; radius R = {R} m")
print(f"fan of the GT mask: {fan.mean():.1%} of the grid")
for th in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
    m = p >= th
    print(f"  p >= {th:4.2f}: {m.mean():5.1%} of the grid, {(m & fan).sum() / max(m.sum(), 1):4.0%} of it inside the fan")
print("share of cells measured in >= 10 % of the keyframes, by range (x forward, |y| < 26):")
for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
    s = (gx >= a) & (gx < b)
    print(f"  x {a:2d}-{b:2d} m: p >= 0.10 on {(p[s] >= 0.10).mean():4.0%} of the cells, mean p {p[s].mean():.2f}")
fig, ax = plt.subplots(1, 4, figsize=(17, 5.4))
cm = plt.get_cmap("viridis")
view.show(ax[0], np.stack([fan] * 3, -1).astype(np.float32) * 0.8, f"GT radar mask today: the fan, {fan.mean():.0%} of the grid")
view.show(ax[1], cm(p)[..., :3].astype(np.float32), f"p(measured within {R:g} m), {n} mini keyframes")
for a_, th in zip(ax[2:], (0.05, 0.20)):
    m = p >= th
    view.show(a_, np.stack([m] * 3, -1).astype(np.float32) * 0.8, f"p >= {th:g}: {m.mean():.0%} of the grid")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radar_density.png")
fig.savefig(out, dpi=80)
print("saved", out)

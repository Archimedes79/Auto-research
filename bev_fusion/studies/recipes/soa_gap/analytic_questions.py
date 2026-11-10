"""Analytic questions on why GaussianCaR beats our system on the maps (2026-10-30). Pure arithmetic on the camera model of nuScenes
(calibration read from v1.0-mini: the rig is the same in every scene up to small differences), our image pipeline
(nuscenes_front_ida: 1600 x 900 -> 704 wide, bottom 256 rows kept, feature stride 16) and GaussianCaR's (448 x 800, stride 8, 128 channels,
numbers from the paper note), plus the tolerance curves of temp/soa_gap/map_ref_r4soa_400.json. CPU, one thread."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "training")
from nuscenes.nuscenes import NuScenes  # noqa: E402

nusc = NuScenes("v1.0-mini", dataroot=os.path.expanduser("~/data/nuscenes"), verbose=False)
s = nusc.sample[0]


def cal(ch):
    sd = nusc.get("sample_data", s["data"][ch])
    c = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return np.array(c["camera_intrinsic"]), np.array(c["translation"]), np.array(c["rotation"])


K, t, q = cal("CAM_FRONT")
f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
H_CAM = float(t[2])
print(f"CAM_FRONT: f = {f:.1f} px, cx = {cx:.1f}, cy = {cy:.1f} (1600 x 900), camera height {H_CAM:.2f} m")

# ---------------------------------------------------------------- Q2 geometry of one feature cell on the ground plane
ours = dict(name="ours (256 x 704, stride 16)", scale=704 / 1600, stride=16, top=900 * (704 / 1600) - 256)
soa = dict(name="GaussianCaR (448 x 800, stride 8)", scale=800 / 1600, stride=8, top=0.0)


def ground_row(sysd, d):
    """image row (in the system's input image) of the ground point at distance d (pitch 0)"""
    return (cy * sysd["scale"] - sysd["top"]) + f * sysd["scale"] * H_CAM / d


def depth_of_row(sysd, v):
    return f * sysd["scale"] * H_CAM / (v - (cy * sysd["scale"] - sysd["top"]))


print("\nQ2  ground footprint of ONE feature cell (cross-range width, down-range depth) and a 15 cm marking in input pixels")
print("%-6s | %-34s | %-34s" % ("d [m]", ours["name"], soa["name"]))
for d in (10, 20, 30, 40, 50, 60):
    row = []
    for sd in (ours, soa):
        fs = f * sd["scale"]
        width = sd["stride"] * d / fs
        v0 = ground_row(sd, d)
        depth = depth_of_row(sd, v0 - sd["stride"] / 2) - depth_of_row(sd, v0 + sd["stride"] / 2)
        row.append("%.2f m wide, %5.1f m deep, line %.1f px" % (width, depth, fs * 0.15 / d))
    print("%-6d | %-34s | %-34s" % (d, *row))

# ---------------------------------------------------------------- Q1 information entering the BEV per BEV cell
cells_ours = (256 // 16) * (704 // 16)
cells_soa = (448 // 8) * (800 // 8)
vals_ours, vals_soa = cells_ours * 80, cells_soa * 128
bev_cells = 120 * 104
print(f"\nQ1  image values handed to the BEV by the front camera: ours {cells_ours} cells x 80 ch = {vals_ours}; GaussianCaR {cells_soa} cells x 128 ch = {vals_soa} "
      f"-> {vals_soa / vals_ours:.1f} x more values, {cells_soa / cells_ours:.1f} x more cells")
print(f"    per BEV cell of our 120 x 104 front grid: ours {vals_ours / bev_cells:.1f}, GaussianCaR {vals_soa / bev_cells:.1f}")

# ---------------------------------------------------------------- Q4 training budget in sample passes
ours_pass = (7000 * 6, 7000 * 18)         # clips (report: 2.6 epochs) .. keyframes seen through the clips
soa_pass = 40 * 28130
print(f"\nQ4  sample passes: ours {ours_pass[0]} (clips) .. {ours_pass[1]} (keyframes through the clips) = {ours_pass[0] / 16289:.1f} .. {ours_pass[1] / 16289:.1f} epochs of 16289 train keyframes; "
      f"GaussianCaR 40 epochs x 28130 = {soa_pass} -> {soa_pass / ours_pass[1]:.0f} .. {soa_pass / ours_pass[0]:.0f} x more")

# ---------------------------------------------------------------- Q8 which cells of our evaluation does a side camera also see?
def yaw_of(qv):
    w, x, y, z = qv
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


half = math.atan(800 / f)
xs = np.arange(0.25, 60, 0.5)
ys = np.arange(-25.75, 26, 0.5)
X, Y = np.meshgrid(xs, ys, indexing="ij")
front = (np.abs(np.arctan2(Y, X)) <= half)
side_seen = np.zeros_like(front)
for ch in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
    Ks, ts, qs = cal(ch)
    yaw = yaw_of(qs)  # camera optical axis yaw in the ego frame is yaw of the rotation about z; the optical axis is the z axis of the camera
    # the optical axis direction in ego: rotate (0, 0, 1) by q
    from pyquaternion import Quaternion
    ax = Quaternion(qs).rotate(np.array([0.0, 0.0, 1.0]))
    az = math.atan2(ax[1], ax[0])
    hs = math.atan(Ks[0, 2] / Ks[0, 0])
    ang = np.arctan2(Y - ts[1], X - ts[0]) - az
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    side_seen |= (np.abs(ang) <= hs)
print(f"\nQ8  cells of the front camera cone (|azimuth| <= {math.degrees(half):.1f} deg, x < 60 m) that a side camera also sees: {100 * (front & side_seen).sum() / front.sum():.1f} %")

# ---------------------------------------------------------------- Q6 position or presence? tolerance curves
d = json.load(open("temp/soa_gap/map_ref_r4soa_400.json"))["systems"]
print("\nQ6  ratio ours / GaussianCaR of the IoU at tolerance 0, 0.5, 1 m (threshold 0.5, all ranges): a positional (blur) deficit would shrink toward 1 with the tolerance")
for L in ("drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider"):
    r = [d["ours"][L]["all"]["0.5"][k] / d["gcar"][L]["all"]["0.5"][k] for k in ("iou@0", "iou@0.5", "iou@1")]
    a = [d["ours"][L]["all"]["0.5"][k] for k in ("iou@0", "iou@0.5", "iou@1")]
    g = [d["gcar"][L]["all"]["0.5"][k] for k in ("iou@0", "iou@0.5", "iou@1")]
    print("   %-14s ours %.3f/%.3f/%.3f  SOA %.3f/%.3f/%.3f  ratio %.2f / %.2f / %.2f   missing IoU at 1 m: %.3f" % (L, *a, *g, *r, g[2] - a[2]))

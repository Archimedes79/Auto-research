"""One-off: CAM_FRONT intrinsics / mounting from nuScenes calibration, and what one image row covers on the ground.

Run from the problem folder: wsl python3 temp/loop19/cam_front_geometry.py
"""
import json
import math
import os

root = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
sensors = {s["token"]: s["channel"] for s in json.load(open(os.path.join(root, "sensor.json")))}
cal = [c for c in json.load(open(os.path.join(root, "calibrated_sensor.json"))) if sensors[c["sensor_token"]] == "CAM_FRONT"]
fx = [c["camera_intrinsic"][0][0] for c in cal]
h = [c["translation"][2] for c in cal]
print(f"CAM_FRONT calibrations: {len(cal)}; fx {min(fx):.0f}..{max(fx):.0f}; height {min(h):.2f}..{max(h):.2f} m")


def pitch_deg(q):
    w, x, y, z = q
    # camera z axis (optical axis) in the ego frame: third column of R
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.atan2(zz, math.hypot(zx, zy)))


def roll_yaw_deg(q):
    w, x, y, z = q
    # camera x axis (image right) in the ego frame: first column of R; its z component is the roll
    xz = 2 * (x * z - w * y)
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    return math.degrees(math.asin(xz)), math.degrees(math.atan2(zy, zx))


ry = [roll_yaw_deg(c["rotation"]) for c in cal]
print(f"roll {min(r for r, _ in ry):.2f}..{max(r for r, _ in ry):.2f} deg; yaw {min(y for _, y in ry):.2f}..{max(y for _, y in ry):.2f} deg")
worst = max(abs(r) for r, _ in ry)
print(f"column drift over the full image height at the worst roll: {math.tan(math.radians(worst)) * 900:.1f} px (full res)")
p = [pitch_deg(c["rotation"]) for c in cal]
print(f"optical axis elevation over the ego xy plane: {min(p):.2f}..{max(p):.2f} deg (negative = looking down)")
f, hc = sum(fx) / len(fx), sum(h) / len(h)
print("range d | metres of ground per image row (full res) | px per 0.5 m cell laterally (full res)")
for d in (5, 10, 20, 35, 50, 60):
    print(f"{d:5d} m | {d * d / (f * hc):6.2f} m/row | {0.5 * f / d:6.1f} px")

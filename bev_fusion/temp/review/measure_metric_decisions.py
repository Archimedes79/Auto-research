"""One-off of review pass 2 (2026-10-31): how big are the metric questions of D29-D32 (C5 field of view, C6 detection wedge,
C7 moving threshold, the zc F1)? Frames: the 1000 val frames of the t8_ctl_s0 evaluation (its frames.jsonl); GT: the stored
keyframes of signature 15efbde2d1 (the GT of that evaluation and of tonight's t7 re-evaluations).
  C5  cells of the evaluation's camera mask (full 1600 x 900 image, z > 0.5 m) that the BEVFusion test crop cannot see (resize 0.48,
      crop x 32-736 / y 176-432 = original u 66.7-1533.3, v 366.7-900): all field-of-view cells, GT drivable, GT lane divider
  C6  scored GT boxes (observed, centre 0 <= x < 60 m inside +-32.3 deg: metrics.in_front) whose centre is outside the grid (|y| >= 26 m)
  C7  scored GT boxes inside the grid with a known speed: > 0.5 m/s (the metric's "moving"), > 1.0 m/s (autogt's checks), in between
  zc  lane_divider:zc / road_divider:zc (f1, p, r at 0.5 m, dt 0) and the vehicle static-velocity numbers of every evaluated run
Run from bev_fusion/ (CPU, 2 threads):
  OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/review/measure_metric_decisions.py > temp/review/measure_metric_decisions.txt
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from autogt import suite                                         # noqa: E402
from autogt.cli import load_keyframe                             # noqa: E402
from autogt.grid import GridSpec                                 # noqa: E402
from autogt.layers import cell_centres                           # noqa: E402
import metrics                                                   # noqa: E402  evaluation/metrics.py

SIG = "15efbde2d1"
GROUPS = ("vehicle", "two_wheeler", "pedestrian")
STORE = os.path.join(ROOT, "runs", "t8_ctl_s0", "eval", "suite", "quant", "eval_n1000")
CROP = (32 / 0.48, 736 / 0.48, 176 / 0.48, 900.0)                # BEVFusion test crop in original pixels (u0, u1, v0, v1)


def cam_mask(meta, gx, gy, u0, u1, v0, v1):
    """ground cells (z = 0 in the ego frame) that project into [u0, u1) x [v0, v1) of the camera, in front of it (as nuscenes_gt)."""
    K, T = np.asarray(meta["K"]), np.asarray(meta["T_c2e"])
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(T).T
    z = P[:, 2]
    u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return ((z > 0.5) & (u >= u0) & (u < u1) & (v >= v0) & (v < v1)).reshape(gx.shape)


def pct(a, b):
    return f"{a} of {b} ({100.0 * a / max(b, 1):.1f} %)"


def main():
    frames = [json.loads(line) for line in open(os.path.join(STORE, "frames.jsonl"))]
    sig_dir = os.path.join(ROOT, "data", "autogt", SIG)
    grid = GridSpec()
    gx, gy = cell_centres(grid)
    c = dict(frames=0, mismatch=0, fov=0, fov_lost=0, div=0, div_lost=0, drv=0, drv_lost=0)
    box = {g: dict(wedge=0, outside=0, far=0, far_outside=0, known=0, mov05=0, mov10=0) for g in GROUPS}
    for f in frames:
        g = load_keyframe(sig_dir, f["key"])
        meta = g["meta"]
        stored = np.asarray(g["camera"]) > 0
        full = cam_mask(meta, gx, gy, 0.0, 1600.0, 0.0, 900.0)
        crop = cam_mask(meta, gx, gy, *CROP)
        c["mismatch"] += int((stored != full).sum())
        lost = stored & ~crop
        div, drv = np.asarray(g["lane_divider"]) > 0.5, np.asarray(g["drivable_area"]) > 0.5
        c["frames"] += 1
        c["fov"] += int(stored.sum()); c["fov_lost"] += int(lost.sum())
        c["div"] += int((div & stored).sum()); c["div_lost"] += int((div & lost).sum())
        c["drv"] += int((drv & stored).sum()); c["drv_lost"] += int((drv & lost).sum())
        for b in metrics.gt_boxes_from_meta(meta["boxes"]):
            if b["group"] not in box or not metrics.in_front(b):
                continue
            s = box[b["group"]]
            out, far = abs(b["cy"]) >= grid.y_max, b["cx"] >= 40.0
            s["wedge"] += 1; s["outside"] += out; s["far"] += far; s["far_outside"] += far and out
            sp = math.hypot(b["vx"], b["vy"])
            if not out and math.isfinite(sp):
                s["known"] += 1; s["mov05"] += sp > metrics.MOVING_SPEED; s["mov10"] += sp > 1.0

    print(f"frames: {c['frames']} val frames of the t8_ctl_s0 evaluation (eval_n1000), GT {SIG} "
          f"(the live autogt signature now: {suite.current_signature()}); grid x {grid.x_min}..{grid.x_max} m, y {grid.y_min}..{grid.y_max} m, "
          f"{grid.res} m cells")
    print(f"check: recomputed full-image camera mask vs the stored 'camera' layer: {c['mismatch']} cells differ")
    print("C5 cells of the evaluation's field of view (full image) that the BEVFusion crop cannot see:")
    print(f"   all field-of-view cells   {pct(c['fov_lost'], c['fov'])}")
    print(f"   GT drivable cells         {pct(c['drv_lost'], c['drv'])}")
    print(f"   GT lane-divider cells     {pct(c['div_lost'], c['div'])}")
    tot = {k: sum(box[g][k] for g in GROUPS) for k in box["vehicle"]}
    print("C6 scored GT boxes (observed, in the +-32.3 deg wedge, 0 <= x < 60 m) whose centre lies outside the grid (|y| >= 26 m):")
    for name, s in list(box.items()) + [("all three", tot)]:
        print(f"   {name:12s} {pct(s['outside'], s['wedge'])}; of those at x >= 40 m: {pct(s['far_outside'], s['far'])}")
    print("C7 scored GT boxes inside the grid with a known speed: moving at > 0.5 m/s (the metric now) / > 1.0 m/s (autogt's checks):")
    for name, s in list(box.items()) + [("all three", tot)]:
        print(f"   {name:12s} known {s['known']}: > 0.5 m/s {s['mov05']}, > 1.0 m/s {s['mov10']}, between 0.5 and 1.0 m/s {s['mov05'] - s['mov10']}")

    print("zc + static velocity of every evaluated run (runs/*/eval/metrics.json; dt 0; vehicle velocity split only in runs evaluated "
          "after 2026-10-31 evening):")
    print(f"   {'run':24s} {'lane_div:zc f1':>14s} {'p':>6s} {'r':>6s}   {'road_div:zc f1':>14s} {'r':>6s}   "
          f"{'ave_static':>10s} {'ave0_static':>11s} {'skill_static':>12s}")
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval", "metrics.json"))):
        m = json.load(open(p)).get("metrics", {})
        z = lambda layer, q: m.get(f"seg_fov/seg/dt+0.00/{layer}:zc/all/all/{q}@0.5", float("nan"))   # noqa: E731
        v = [m.get(f"det/dt+0.00/vehicle/all/{q}", float("nan")) for q in ("ave_static", "ave0_static", "ave_skill_static")]
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        print(f"   {name:24s} {z('lane_divider', 'f1'):14.4f} {z('lane_divider', 'p'):6.3f} {z('lane_divider', 'r'):6.4f}   "
              f"{z('road_divider', 'f1'):14.4f} {z('road_divider', 'r'):6.4f}   {v[0]:10.4f} {v[1]:11.4f} {v[2]:12.4f}")


if __name__ == "__main__":
    main()

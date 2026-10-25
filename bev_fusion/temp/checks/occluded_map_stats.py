"""How much of the map GT is hidden behind objects? (decision 1 of plan 8.4: weight of occluded map cells)

Real nuScenes mini, every 3rd keyframe, GT of training/gt_bev.py on the 0.5 m front grid. Among the cells the front camera reaches
(camera mask): the share occluded (behind a GT box seen from the camera), overall and per range bin, and the same share among the
cells of each map layer (drivable area > 0.5 coverage, lane / road divider cells, centre-line cells); and how much of the occluded
map lies in radar coverage.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/occluded_map_stats.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "training")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402

grid = Grid()
gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
xs = grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res
ys = grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res
rng = np.hypot(xs[:, None], ys[None, :])
bins = [("0-20 m", 0, 20), ("20-40 m", 20, 40), ("40-60 m", 40, 61)]
tot = {}


def add(name, mask, occ, rad):
    t = tot.setdefault(name, np.zeros(3))
    t += [mask.sum(), (mask & occ).sum(), (mask & occ & rad).sum()]


samples = gt.nusc.sample[::3]
for k, s in enumerate(samples):
    g = gt.keyframe_gt(s["token"], grid)
    cam, occ, rad = g["camera"] > 0, g["occluded"] > 0, g["radar"] > 0
    for label, lo, hi in bins:
        b = (rng >= lo) & (rng < hi)
        add(f"all camera cells {label}", cam & b, occ, rad)
        add(f"drivable area {label}", cam & b & (g["drivable_area"] > 0.5), occ, rad)
        add(f"lane divider {label}", cam & b & (g["lane_divider"] > 0), occ, rad)
        add(f"centre-line {label}", cam & b & (g["centreline"] > 0), occ, rad)
    add("all camera cells", cam, occ, rad)
    add("drivable area", cam & (g["drivable_area"] > 0.5), occ, rad)
    add("lane divider", cam & (g["lane_divider"] > 0), occ, rad)
    add("road divider", cam & (g["road_divider"] > 0), occ, rad)
    add("centre-line", cam & (g["centreline"] > 0), occ, rad)
print(f"{len(samples)} keyframes; share of the cells the camera reaches that are occluded (behind a box), and of those the share inside radar coverage")
print(f"{'':32s}{'cells / frame':>14s}{'occluded':>10s}{'of which in radar cov.':>24s}")
for name, t in tot.items():
    print(f"{name:32s}{t[0] / len(samples):14.0f}{100 * t[1] / max(t[0], 1):9.1f}%{100 * t[2] / max(t[1], 1):23.1f}%")

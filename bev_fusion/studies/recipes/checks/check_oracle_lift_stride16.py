"""One-off: the oracle lift (tests/test_oracle_lift.py) at the REAL encoder's resolution - DualBEV's features are at
stride 16 of the 256 x 704 input (16 x 44), the gate test runs at stride 8 (32 x 88). Same 6 scenes, same scoring.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_oracle_lift_stride16.py"""
import random
import sys

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
sys.path.insert(0, "tests/support")
from lift_pd import Grid  # noqa: E402
from synthetic_world import camera, front_camera, random_scene  # noqa: E402
from test_oracle_lift import NAMES, depth_dist, features, hit_cells, lifted, range_recall, score  # noqa: E402

torch.set_grad_enabled(False)
grid = Grid()
K, T, hw = front_camera()
Z0 = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
variants = {"Grid slices, min depth": (grid.z_levels, 1, "min"),
            "Grid slices + oversampling x4, histogram": (grid.z_levels, 4, "hist"),
            "z = 0 slices + oversampling x4, histogram": (Z0, 4, "hist"),
            "z = 0 slices, histogram": (Z0, 1, "hist")}
rng = random.Random(7)
acc = {(v, s): [] for v in variants for s in (8, 16)}
lr = {(v, s): [] for v in variants for s in (8, 16)}
for n in range(6):
    sc = random_scene(rng)
    img, depth, cls = camera(sc, K, T, hw)
    hits = hit_cells(depth, cls, sc, K, T, grid)
    for s in (8, 16):
        feat = features(sc, cls, s)
        for name, (zl, sub, mode) in variants.items():
            lf = lifted(feat, depth_dist(depth, s, mode), K, T, hw, grid, zl, sub)
            acc[(name, s)].append(score(lf, hits))
            lr[(name, s)].append(range_recall(lf, hits, 1))
print("variant | stride | recall / precision: " + " | ".join(NAMES) + " || lane recall 0-20 / 20-40 / 40-60 m")
for (name, s), runs in acc.items():
    cols = [f"{sum(r[j][0] for r in runs) / 6:.2f} / {sum(r[j][1] for r in runs) / 6:.2f}" for j in range(4)]
    bands = " / ".join(f"{sum(x[i] for x in lr[(name, s)]) / 6:.2f}" for i in range(3))
    print(f"{name:42s} | {s:2d} | " + " | ".join(cols) + f" || {bands}")

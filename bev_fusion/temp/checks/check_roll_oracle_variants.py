"""Why does the oracle lift lose lane recall with a rolled camera (tests/test_projection_general.py B)? Variants at roll 0 / 20 deg: feature stride 1 / 8,
oversampling 1 / 4, depth histogram vs minimum depth. Same 4 scenes. Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_roll_oracle_variants.py"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid  # noqa: E402
from synthetic_world import camera, random_scene  # noqa: E402
from test_oracle_lift import depth_dist, features, hit_cells, lifted, score  # noqa: E402
from test_projection_general import make_camera  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(2)
grid = Grid()
hw = (256, 704)
print("roll | stride | oversample | depth | lane recall / precision | vehicle recall / precision")
for roll in (0.0, 20.0):
    K, T = make_camera(roll)
    Kt, Tt = torch.tensor(K), torch.tensor(T)
    rng = random.Random(7)
    scenes = [random_scene(rng) for _ in range(4)]
    rend = [camera(sc, Kt, Tt, hw) for sc in scenes]
    for stride in (1, 8):
        for sub in (1, 4):
            for mode in ("hist", "min"):
                if stride == 1 and mode == "min":
                    continue
                acc = []
                for sc, (img, depth, cls) in zip(scenes, rend):
                    hits = hit_cells(depth, cls, sc, Kt, Tt, grid)
                    lf = lifted(features(sc, cls, stride), depth_dist(depth, stride, mode), Kt, Tt, hw, grid, grid.z_levels, sub)
                    acc.append(score(lf, hits))
                m = lambda j, k: float(np.mean([a[j][k] for a in acc]))      # noqa: E731
                print(f"{roll:4.0f} | {stride:6d} | {sub:10d} | {mode:5s} | {m(1, 0):.2f} / {m(1, 1):.2f} | {m(2, 0):.2f} / {m(2, 1):.2f}", flush=True)

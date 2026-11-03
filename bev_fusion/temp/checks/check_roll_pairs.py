"""Pose-matched pairs for the rolled-camera oracle: the same pose (pitch / yaw / position) with roll 0 and with a roll, 8 scenes, the recommended lift
(stride 8, oversample 4, depth histogram): does the roll itself cost recall, or does the pose? Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_roll_pairs.py"""
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
import torch.nn.functional as F  # noqa: E402


def rprec(lift, hit):
    """threshold-free: of the K = |hit cells| cells with the most lifted mass, the share within one cell of a hit cell."""
    k = int(hit.sum())
    if k == 0:
        return float("nan")
    top = torch.topk(lift.flatten(), k).indices
    near = F.max_pool2d(hit[None, None].float(), 3, 1, 1)[0, 0].flatten() > 0
    return float(near[top].float().mean())


torch.set_grad_enabled(False)
torch.set_num_threads(2)
grid, hw = Grid(), (256, 704)
poses = [dict(), dict(pitch=6.0, yaw=8.0, pos=(0.3, 0.4, 1.9))]
for pose in poses:
    for roll in (0.0, 20.0, -25.0, 35.0):
        K, T = make_camera(roll, **pose)
        Kt, Tt = torch.tensor(K), torch.tensor(T)
        rng = random.Random(11)
        acc, rp = [], []
        for _ in range(8):
            sc = random_scene(rng)
            img, depth, cls = camera(sc, Kt, Tt, hw)
            hits = hit_cells(depth, cls, sc, Kt, Tt, grid)
            lf = lifted(features(sc, cls, 8), depth_dist(depth, 8, "hist"), Kt, Tt, hw, grid, grid.z_levels, 4)
            acc.append(score(lf, hits))
            rp.append([rprec(lf[1], hits[1]), rprec(lf[2], hits[2])])
        m = lambda j, k: float(np.mean([a[j][k] for a in acc]))      # noqa: E731
        print(f"pose {pose or 'level'} roll {roll:5.0f}: lanes {m(1, 0):.2f} / {m(1, 1):.2f}, vehicles {m(2, 0):.2f} / {m(2, 1):.2f} | R-precision lanes {np.nanmean([r[0] for r in rp]):.2f}, vehicles {np.nanmean([r[1] for r in rp]):.2f}", flush=True)

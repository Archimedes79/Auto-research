"""t21 scratch: position error (in cells) of ONE composed (dt, v, omega) triple (train.gap_ego_motion) against the per-sweep chain, turning ego."""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
import splat as S  # noqa: E402
import train as T  # noqa: E402
from lift_pd import Grid  # noqa: E402

grid = Grid()
p = S.grid_points(grid, dtype=torch.float64)[None]
for n, label in ((2, "2 sweeps (0.167 s, one stride-2 frame)"), (4, "4 sweeps"), (6, "6 sweeps (0.5 s, keyframe gap)")):
    for v0, w0 in ((10.0, 0.0), (10.0, 0.2), (10.0, 0.5), (15.0, -0.8), (5.0, 1.2)):
        chain = [(1 / 12, v0 + 0.3 * i, w0 * (1 + 0.1 * i)) for i in range(n)]
        trip = T.gap_ego_motion(chain)
        pc = S.chain_positions(p, chain)
        p1 = S.splat_positions(p, *trip)
        err = (pc - p1).norm(dim=-1) / grid.res
        print(f"{label:40s} v {v0:5.1f} omega {w0:5.2f}: max {err.max():.4f} cells, mean {err.mean():.4f}")

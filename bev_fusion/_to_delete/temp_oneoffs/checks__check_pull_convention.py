"""One-off: which sign / axis convention of the pull in tests/test_flow_correlation.py matches the data best?
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_pull_convention.py"""
import random
import sys
import time

import torch

sys.path.insert(0, "tests")
from test_flow_correlation import batch, make_pair, pull  # noqa: E402

t0 = time.time()
pairs = [make_pair(random.Random(99990 + i)) for i in range(12)]
print(f"12 pairs in {time.time() - t0:.0f} s")
b = batch(pairs)
m = b["moving"].unsqueeze(1).float()
edge = m * ((b["f0"] - b["f1_in0"]).abs().amax(1, keepdim=True) > 0.05).float()
v = b["vel"]
for name, vv in {"v": v, "-v": -v, "(vx,-vy)": torch.stack([v[:, 0], -v[:, 1]], 1),
                 "(-vx,vy)": torch.stack([-v[:, 0], v[:, 1]], 1), "swap": v.flip(1), "0": 0 * v,
                 "v/2": v / 2, "2v": 2 * v}.items():
    d = (pull(b["f1_in0"], vv) - b["f0"]).abs()
    print(f"{name:9s} moving cells {(d * m).sum() / m.sum():.3f}   changed moving cells {(d * edge).sum() / edge.sum():.3f}")

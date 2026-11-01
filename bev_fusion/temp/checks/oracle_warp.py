"""Is the warp loss minimised by the true motion? Warp MSE of the TRUE velocity field (and of v = 0) on the held-out
scenes of tests/test_flow_refine.py, against the learned arms (known cells, same definition as the test)."""
import random, sys
import torch
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_refine as T
from temporal import propagate
test = [T.sample(random.Random(10_000 + i)) for i in range(12)]
for name, fn in (("true velocity", lambda s: T.true_flow(s)), ("zero velocity", lambda s: torch.zeros(2, T.G.nx, T.G.ny))):
    se, n = 0.0, 0
    for src, tgt, steps, hole in test:
        out = propagate(src[None], [steps], fn(src)[None], T.G)[0][0]
        e = ((out - tgt) ** 2).sum(0)
        se += e[~hole].sum().item(); n += int((~hole).sum()) * 3
    print(f"{name:14s} warp MSE known {se / n:.5f}")

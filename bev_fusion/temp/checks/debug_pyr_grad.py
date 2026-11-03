import sys, torch
sys.path.insert(0, "training"); sys.path.insert(0, "tests")
import test_temporal as T
from temporal import FlowPyramid
x = torch.randn(1, 2, T.G.nx, T.G.ny, dtype=torch.float64)
v0 = torch.randn(1, 2, T.G.nx, T.G.ny, dtype=torch.float64)
fpy = FlowPyramid(2, T.G).double()
for lv in fpy.levels:
    for p in lv.net[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
fpy(x, x.roll(2, 2), [(0.5, 4.0, 0.1)], v0)[-1].pow(2).sum().backward()
for n, p in fpy.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        print("no grad:", n, None if p.grad is None else 0)

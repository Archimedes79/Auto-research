import sys, torch
sys.path.insert(0, "training"); sys.path.insert(0, "tests")
import test_temporal as T
from temporal import FlowPyramid
x = torch.randn(1, 2, T.G.nx, T.G.ny, dtype=torch.float64)
v0 = torch.randn(1, 2, T.G.nx, T.G.ny, dtype=torch.float64, requires_grad=True)
fpy = FlowPyramid(2, T.G).double()
for lv in fpy.levels:
    for p in lv.net[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
sum(v.pow(2).sum() for v in fpy(x, x.roll(2, 2), [(0.5, 4.0, 0.1)], v0)).backward()
print("gradient into v0 from the pyramid outputs:", None if v0.grad is None else v0.grad.abs().sum().item())

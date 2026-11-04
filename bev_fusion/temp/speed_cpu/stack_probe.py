import torch
from torch.profiler import ProfilerActivity, profile


def f(x):
    m = x > 0
    return x[m].sum().item()


with profile(activities=[ProfilerActivity.CPU], with_stack=True) as p:
    f(torch.randn(100))
for e in p.events():
    if e.name in ("aten::item", "aten::nonzero"):
        print(e.name, repr(e.stack)[:300])

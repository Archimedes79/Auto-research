"""time of the box extraction on head maps of the real size (B = 1, 3 groups, 120 x 104 cells), CPU, median of 200 calls.
decode      = object_head.decode on float32 CPU maps (what the evaluation adapter calls)
model_boxes = what BEVFusionModel.boxes does: maps -> .double().cpu() -> dense_boxes -> select_boxes
fast        = extract.boxes_from_maps (after the change) on the same float32 maps"""
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
torch.set_num_threads(2)
import extract as E                                             # noqa: E402
import object_head as O                                         # noqa: E402
from lift_pd import Grid                                        # noqa: E402
from losses import NUSC_GROUPS                                  # noqa: E402

grid = Grid()
B, G, X, Y = 1, 3, grid.nx, grid.ny
torch.manual_seed(0)
obj = {"hm": torch.randn(B, G, X, Y) * 1.5 - 2.5, "off": torch.rand(B, 2, X, Y), "size": torch.randn(B, 2, X, Y) * 0.2 + 1.2,
       "rot": torch.randn(B, 2, X, Y), "rad": torch.randn(B, 1, X, Y) * 0.1 - 0.5, "vel": torch.randn(B, 2, X, Y),
       "dir": torch.randn(B, 1, X, Y), "mov": torch.randn(B, 1, X, Y)}


def t(fn, n=200):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts)


kw = dict(k=100, threshold=0.05)
print("candidates above the threshold:", len(O.decode(obj, grid, NUSC_GROUPS, **kw)[0]))
print("decode      %.2f ms" % t(lambda: O.decode(obj, grid, NUSC_GROUPS, **kw)))
print("model_boxes %.2f ms" % t(lambda: E.select_boxes(E.dense_boxes({k: v.double().cpu() for k, v in obj.items()}, grid, NUSC_GROUPS, soft=False),
                                                        NUSC_GROUPS, **kw)))
if hasattr(E, "boxes_from_maps"):
    print("fast        %.2f ms" % t(lambda: E.boxes_from_maps(obj, grid, NUSC_GROUPS, **kw)))

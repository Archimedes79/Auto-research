import os, sys, time
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
for flag in (False, True):
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = flag
    ds = LD.QData("val_clean", map_mode="exact", pixels=False)
    t0 = time.time()
    for i in range(10):
        o = ds[i]
    dt = (time.time() - t0) / 10
    print(f"pscale {flag}: {1000*dt:.0f} ms per item; pscale finite entries {int(torch.isfinite(o['pscale']).sum())} of {o['pscale'].numel()}")

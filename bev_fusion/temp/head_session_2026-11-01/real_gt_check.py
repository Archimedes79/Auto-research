"""fusion_area_only on REAL GT frames (data/cache/gt, signature 15efbde2d1): share of the grid removed, objects dropped, default-off identity."""
import glob
import os
import sys
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(ROOT))
import losses as L                                              # noqa: E402
from lift_pd import Grid                                        # noqa: E402

torch.set_num_threads(2)
files = sorted(glob.glob(os.path.join(ROOT, "data/cache/gt/*_0.5_4_15efbde2d1.npz")))[::97][:40]
grid, cfg = Grid(), L.LossConfig()
gts = [np.load(f, allow_pickle=True)["gt"].item()["cur"] for f in files]
print("frames", len(gts), "keys", sorted(k for k in gts[0] if k != "meta")[:30])
print("meta keys", sorted(gts[0]["meta"].keys()))
off = L.build_layer_targets(gts, cfg)
on = L.build_layer_targets(gts, replace(cfg, fusion_area_only=True))
fa = off["w_fusion"]
print("fusion area share of the grid: %.3f (camera %.3f, radar %.3f)" % (float(fa.mean()), float(off["w_cam"].mean() + 0), float(off["w_radar"].mean())))
for k in ("w_map", "w_lines", "w_obj"):
    print(f"{k}: mean off {float(off[k].mean()):.4f} on {float(on[k].mean()):.4f}; changed only outside fusion area:",
          bool(torch.equal(on[k], off[k] * fa)))
for k in ("w_cam", "w_radar", "w_keep"):
    assert torch.equal(on[k], off[k]), k
# objects: how many boxes are supervised (centre cell weight > 0), off vs on
n = {}
for tag, T in (("off", off), ("on", on)):
    c = 0
    for b, boxes in enumerate(T["boxes"]):
        for bx in boxes:
            hb = L.head_box(bx)
            if hb is None:
                continue
            ix = int(np.floor((hb["cx"] - grid.x_min) / grid.res))
            iy = int(np.floor((hb["cy"] - grid.y_min) / grid.res))
            if 0 <= ix < grid.nx and 0 <= iy < grid.ny and bx.get("observed", True) and float(T["w_obj"][b, 0, ix, iy]) > 0:
                c += 1
    n[tag] = c
print("supervised object centres: off", n["off"], "on", n["on"], "(%.1f %% dropped)" % (100 * (1 - n["on"] / max(n["off"], 1))))

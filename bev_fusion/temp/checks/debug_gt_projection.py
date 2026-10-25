"""One-off: where do near-range GT divider cells of scene-0061 project in the front image? (projection sanity check)"""
import os
import sys

import numpy as np

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from lift_pd import Grid  # noqa: E402
from view_training_samples import project  # noqa: E402

G = Grid()
src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
sc = next(s for s in src.nusc.scene if s["name"] == "scene-0061")
tok = sc["first_sample_token"]
for _ in range(8):
    tok = src.nusc.get("sample", tok)["next"]
gt = src.keyframe_gt(tok, G)
gx, gy = cell_centres(G)
print("camera T_c2e translation", gt["meta"]["T_c2e"][:3, 3], "K fx", gt["meta"]["K"][0, 0])
for layer in ("lane_divider", "road_divider", "centreline", "drivable_area"):
    m = gt[layer] > 0.5
    near = m & (gx > 8) & (gx < 25)
    print(layer, "cells", int(m.sum()), "near (<25 m)", int(near.sum()))
    if near.sum():
        xyz = np.stack([gx[near], gy[near], np.zeros(near.sum())], 1)
        uv, z = project(gt, xyz)
        idx = np.argsort(gx[near])[:5]
        for i in idx:
            print(f"   x {gx[near][i]:5.2f} y {gy[near][i]:6.2f} -> u {uv[i, 0]:7.1f} v {uv[i, 1]:6.1f} z {z[i]:5.2f}")

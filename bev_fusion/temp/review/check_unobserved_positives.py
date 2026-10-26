"""Review check: how many object-head positives (and how much dense 'background') come from boxes WITHOUT any lidar /
radar point, with the current gt_bev + losses code? 12 mid-scene keyframes of nuScenes mini, 0.5 m grid.

For every box the loss would supervise (losses.object_head_terms rule: group known to the head, centre cell inside the
grid, w_obj > 0 at the centre cell) it is counted whether the box is observed; and for the unobserved ones whether the
dense object layers (semantic head) call the same footprint background (coverage 0) - the contradiction.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/review/check_unobserved_positives.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, "training")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402
from losses import NUSC_GROUPS, build_layer_targets, head_box  # noqa: E402

grid = Grid()
gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
samples = [s for s in gt.nusc.sample if s["prev"] and s["next"]][10::30][:12]
sup_obs = sup_un = dense_bg = 0
for s in samples:
    g = gt.keyframe_gt(s["token"], grid, 4)
    T = build_layer_targets([g], None)
    w = T["w_obj"][0, 0].numpy()
    for bx in g["meta"]["boxes"]:
        hb = head_box(bx, NUSC_GROUPS)
        if hb is None:
            continue
        ix, iy = int(math.floor((hb["cx"] - grid.x_min) / grid.res)), int(math.floor((hb["cy"] - grid.y_min) / grid.res))
        if not (0 <= ix < grid.nx and 0 <= iy < grid.ny) or w[ix, iy] <= 0:
            continue
        if bx["observed"]:
            sup_obs += 1
        else:
            sup_un += 1
            cov = max(float(g[k][ix, iy]) for k in ("vehicle", "pedestrian", "two_wheeler"))
            dense_bg += cov < 0.5
tot = sup_obs + sup_un
print(f"{len(samples)} keyframes: {tot} boxes the object head would supervise, {sup_un} of them ({sup_un / max(1, tot):.1%}) "
      f"have no lidar / radar point; for {dense_bg} of those the dense object layer says background at the box centre "
      f"(the contradiction); `ignore` is not read by losses.py: {'ignore' not in open('training/losses.py').read().replace('ignored', '')}")

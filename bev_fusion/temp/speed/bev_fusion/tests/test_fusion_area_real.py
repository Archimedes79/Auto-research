"""LossConfig.fusion_area_only (D38 (b)) on REAL cached GT frames: how much supervision the option removes, and that it removes nothing else.

    python tests/test_fusion_area_real.py [--signature 15efbde2d1] [--frames 40]

Reads every 97th file of data/cache/gt/*_<signature>.npz (the 'cur' keyframe), builds the loss weights with the option off and on and prints
the share of the grid that is the fusion area (camera x radar, D37), the mean map and object weights, and the supervised object centres
(unobserved boxes and centres with weight 0 are not supervised). Asserts: the changed weights lie only outside the fusion area and equal the
default weights x the area, the aux weights are untouched, the area is a real part of the grid (0.4 .. 0.8), and the option drops at least
one supervised object centre. Skips (exit 0) when the cache folder or the signature is not there. CPU, a few seconds, no model, no training.
The numbers of 2026-11-01 are in experiments/t14_dontcare/results/fusion_area_real_gt.txt (card section "Measured before the run").
"""
import argparse
import glob
import os
import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(ROOT))
import losses as L                                              # noqa: E402
from lift_pd import Grid                                        # noqa: E402


def supervised_centres(T, grid):
    n = 0
    for b, boxes in enumerate(T["boxes"]):
        for bx in boxes:
            hb = L.head_box(bx)
            if hb is None:
                continue
            ix = int(np.floor((hb["cx"] - grid.x_min) / grid.res))
            iy = int(np.floor((hb["cy"] - grid.y_min) / grid.res))
            if 0 <= ix < grid.nx and 0 <= iy < grid.ny and bx.get("observed", True) and float(T["w_obj"][b, 0, ix, iy]) > 0:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signature", default="15efbde2d1", help="generator signature of the cached GT (the GT of the t10 / t11 chain)")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", f"*_0.5_4_{a.signature}.npz")))[::97][:a.frames]
    if not files:
        print(f"SKIP: no cached GT of signature {a.signature} in data/cache/gt")
        return 0
    torch.set_num_threads(2)
    grid, cfg = Grid(), L.LossConfig()
    gts = [np.load(f, allow_pickle=True)["gt"].item()["cur"] for f in files]
    off, on = L.build_layer_targets(gts, cfg), L.build_layer_targets(gts, replace(cfg, fusion_area_only=True))
    fa = off["w_fusion"]
    print(f"GT signature {a.signature}, {len(gts)} real frames (every 97th file of data/cache/gt)")
    print(f"fusion area (camera x radar): {float(fa.mean()):.3f} of the grid; camera and not occluded {float(off['w_cam'].mean()):.3f}, "
          f"radar {float(off['w_radar'].mean()):.3f}")
    for k in ("w_map", "w_lines", "w_obj"):
        assert torch.equal(on[k], off[k] * fa), f"{k}: the option must multiply the default weight by the fusion area, nothing else"
        print(f"{k}: mean weight off {float(off[k].mean()):.4f} -> on {float(on[k].mean()):.4f}")
    for k in ("w_cam", "w_radar", "w_keep"):
        assert torch.equal(on[k], off[k]), f"{k}: the aux heads keep their own area"
    n_off, n_on = supervised_centres(off, grid), supervised_centres(on, grid)
    print(f"supervised object centres: off {n_off} -> on {n_on} ({100 * (1 - n_on / max(n_off, 1)):.1f} % dropped)")
    assert 0.4 < float(fa.mean()) < 0.8, "the fusion area should be a real part of the grid"
    assert 0 < n_on < n_off, "the option must drop some, not all, supervised object centres"
    print("fusion_area_real passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

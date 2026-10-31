"""t10 gate (layer 0, CPU, a few seconds): do the two feature branches of camera_features "dual" put the same image pixel at the same BEV cell?

tests/test_camera_features.py covers shapes, branch select, gradients and refusals. This adds what it does not: the GEOMETRY of the pair with the
real image transforms of the two caches (DualBEV: nuscenes_front_ida, cell centres at 16 k + 7.5; BEVFusion: bevfusion_ida 0.48 / crop 32, 176,
frustum cells at j (W - 1) / (w - 1)). Each branch gets a feature map that stores, in every cell, the ORIGINAL-image pixel (x, y) of that cell
(ramp), a constant channel, and a uniform depth distribution; the model's own lifts (md.lift, md.lift2, oversample 4) then give at every BEV
voxel the original-image pixel the voxel projects to. It must equal the analytic projection K [R|t] of the voxel and so agree across the two
branches. A silent mix-up (wrong ida, wrong cell convention, a cache made with another crop) shows up as tens of pixels.

 G7a  DualBEV branch  vs analytic projection: max |error| < 0.5 original pixel on the interior cells seen by all sub-heights
 G7b  BEVFusion branch vs analytic projection: same
 G7c  the two branches agree with each other on the common cells (< 1 pixel), at least 3000 cells compared (not vacuous)
 G8   sensitivity: lifting branch 2 with the DualBEV ida (the mix-up) makes the error > 10 pixels; the gate can see it

Run: wsl bash tools/env/_py14.sh temp/t10/gate.py     (CPU; OMP_NUM_THREADS=2 while a training runs)"""
import json
import os
import sys

import torch

torch.set_num_threads(2)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training")]
from bevfusion_camera import bevfusion_ida  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from nuscenes_dataset import DST_HW, nuscenes_front_ida  # noqa: E402

H, W = DST_HW
assert (H, W) == (256, 704), DST_HW
FX, FY, CX, CY = 1266.4, 1266.4, 816.3, 491.5                       # a nuScenes CAM_FRONT-like intrinsic (original 1600 x 900 image)
K = torch.tensor([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
T = torch.eye(4)
T[:3, :3] = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])     # car -> camera: x forward = camera z, y left = -camera x, z up = -camera y
T[:3, 3] = torch.tensor([0.0, 1.5, -1.5])                           # camera at x = 1.5 m, height 1.5 m
CAM_X, CAM_H = 1.5, 1.5

md = CachedBEVFusionModel(ModelConfig(pretrained="", camera_features="dual")).eval()
ida1, ida2 = nuscenes_front_ida(dst_hw=DST_HW), bevfusion_ida()
nz, C = len(md.cfg.z_levels), 80
D = 118


def ramps(h, w, pix_u, pix_v, ida):
    """(1, 80, h, w): channel 0 = original x of every cell, 1 = original y, 2 = 1 (the weight check); pix_u / pix_v: the augmented-image pixel of each cell."""
    ux, vy = (pix_u - ida[0, 2]) / ida[0, 0], (pix_v - ida[1, 2]) / ida[1, 1]
    f = torch.zeros(1, C, h, w)
    f[0, 0], f[0, 1], f[0, 2] = ux.view(1, w).expand(h, w) / 1000.0, vy.view(h, 1).expand(h, w) / 1000.0, 1.0
    return f


def lifted(lift, feat, ida):
    prob = torch.full((1, D) + feat.shape[-2:], 1.0 / D)
    with torch.no_grad():
        vol = lift(feat, prob, K, T, (H, W), ida)[0]                # (nz * 80, X, Y)
    return vol.view(nz, C, *vol.shape[1:]), prob


def decode(vol):
    """per z-level: (x, y, weight) in original pixels; weight 1 = every sub-height of the voxel valid"""
    wgt = vol[:, 2] * D
    ok = wgt > 1.0 - 1e-3
    return vol[:, 0] / vol[:, 2].clamp(min=1e-9) * 1000.0, vol[:, 1] / vol[:, 2].clamp(min=1e-9) * 1000.0, ok


def analytic(grid, z_levels):
    xs, ys = grid.xy_centres(torch.device("cpu"), torch.float32)
    X, Y = len(xs), len(ys)
    gx, gy = xs.view(1, X, 1).expand(len(z_levels), X, Y), ys.view(1, 1, Y).expand(len(z_levels), X, Y)
    gz = torch.tensor(z_levels, dtype=torch.float32).view(-1, 1, 1).expand(len(z_levels), X, Y)
    zc = gx - CAM_X
    return FX * (-gy) / zc + CX, FY * (CAM_H - gz) / zc + CY, zc


# DualBEV branch: 16 x 44 cells, cell k at augmented pixel 16 k + 7.5
h1, w1 = H // 16, W // 16
u1, v1 = torch.arange(w1) * 16 + 7.5, torch.arange(h1) * 16 + 7.5
# BEVFusion branch: 32 x 88 cells, cell j at pixel j (W - 1) / (w - 1)
h2, w2 = 32, 88
u2, v2 = torch.linspace(0, W - 1, w2), torch.linspace(0, H - 1, h2)
f1, f2 = ramps(h1, w1, u1, v1, ida1), ramps(h2, w2, u2, v2, ida2)

ax, ay, zc = analytic(md.grid, md.cfg.z_levels)
res = {}


def compare(lift, feat, ida, interior):
    vol, _ = lifted(lift, feat, ida)
    ex, ey, ok = decode(vol)
    m = ok & (zc > 1.5) & (zc < 55.0)
    # interior: the pixel the voxel projects to lies where the feature map has real cells around it (no border clamp)
    px_aug = (ax * ida[0, 0] + ida[0, 2], ay * ida[1, 1] + ida[1, 2])
    lo_u, hi_u, lo_v, hi_v = interior
    n_os = lift.oversample                                           # the sub-heights of a slice spread its row by fy dz / z_c: all of them must stay inside
    spread = ida[1, 1] * FY * (1 - 1 / n_os) / 2 * lift.slice_height / zc
    m = m & (px_aug[0] > lo_u) & (px_aug[0] < hi_u) & (px_aug[1] - spread > lo_v) & (px_aug[1] + spread < hi_v)
    err = torch.maximum((ex - ax).abs(), (ey - ay).abs())
    return ex, ey, m, float(err[m].max()) if m.any() else float("nan"), int(m.sum())


# the DualBEV map interpolates between cell centres 7.5 .. W - 8.5: stay inside; BEVFusion's cells span the whole image
ex1, ey1, m1, e1, n1 = compare(md.lift, f1, ida1, (8.0, W - 9.0, 8.0, H - 9.0))
ex2, ey2, m2, e2, n2 = compare(md.lift2, f2, ida2, (0.5, W - 1.5, 0.5, H - 1.5))
if os.environ.get("T10_GATE_VERBOSE"):
    for nm, ex_, ey_, m_, ida_ in (("dual", ex1, ey1, m1, ida1), ("bevf", ex2, ey2, m2, ida2)):
        e_x, e_y = (ex_ - ax).abs(), (ey_ - ay).abs()
        k = int(torch.where(m_, torch.maximum(e_x, e_y), torch.zeros(())).flatten().argmax())
        zi, xi, yi = k // (ax.shape[1] * ax.shape[2]), (k // ax.shape[2]) % ax.shape[1], k % ax.shape[2]
        print(nm, "max err x", float(e_x[m_].max()), "y", float(e_y[m_].max()), "mean signed x", float((ex_ - ax)[m_].mean()), "y", float((ey_ - ay)[m_].mean()),
              "worst voxel z/x/y", zi, xi, yi, "car (x,y,z)", float(zc[zi, xi, yi] + CAM_X), "aug px", float(ax[zi, xi, yi] * ida_[0, 0] + ida_[0, 2]),
              float(ay[zi, xi, yi] * ida_[1, 1] + ida_[1, 2]), "est/true", float(ex_[zi, xi, yi]), float(ax[zi, xi, yi]), float(ey_[zi, xi, yi]), float(ay[zi, xi, yi]))
res["G7a"] = {"max_err_px": e1, "cells": n1, "pass": bool(e1 < 0.5 and n1 > 3000)}
res["G7b"] = {"max_err_px": e2, "cells": n2, "pass": bool(e2 < 0.5 and n2 > 3000)}
both = m1 & m2
dx, dy = (ex1 - ex2).abs()[both], (ey1 - ey2).abs()[both]
cross = float(torch.maximum(dx, dy).max()) if both.any() else float("nan")
res["G7c"] = {"max_diff_px": cross, "cells": int(both.sum()), "pass": bool(cross < 1.0 and int(both.sum()) > 3000)}
# sensitivity: branch 2 lifted with the DualBEV ida (the mix-up the gate has to see)
mixed = decode(lifted(md.lift2, f2, ida1)[0])
mixed_err = float(torch.maximum((mixed[0] - ax).abs(), (mixed[1] - ay).abs())[both & mixed[2]].max())
res["G8"] = {"err_with_wrong_ida_px": mixed_err, "pass": bool(mixed_err > 10.0)}
res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(ROOT, "temp", "t10"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "temp", "t10", "gate_result.json"), "w"), indent=1)
assert res["all_pass"], "t10 gate FAILED"
print("t10 gate ok: both branches put the same image pixel at the same BEV cell (real idas), the mix-up is visible")

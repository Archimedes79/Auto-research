"""T0 (scout, 2026-10-26): how much of what a radar sees of the STATIONARY world does the proposed proxy of plan 8.4 explain?

Radar: ego-aligned detections of the last 0.5 s (cache temp/radar/cache/mini_multisweep.pt, in the ego frame of the CAM_FRONT time
of each keyframe = the frame of the GT). Static returns: |ego-compensated radial velocity| < 0.5 m/s, inside the front grid.
Proxy C: footprints of stationary annotated boxes (speed < 0.5 m/s) + a 1 m band along the edge of the drivable area.
Reported per range bin: (recall) the share of static returns within one cell of the proxy; (precision) the share of proxy cells with a
static return within one cell; the same for a NULL (the proxy shifted 5 m sideways, chance rate); where the static returns are
(category of their cell) and the distance of static returns from the drivable area.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/t0_stationary_proxy.py [every-nth keyframe]
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "training")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402

step = int(sys.argv[1]) if len(sys.argv) > 1 else 7
grid = Grid()
gt = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
frames = torch.load("temp/radar/cache/mini_multisweep.pt")
xs = grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res
ys = grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res
rng = np.hypot(xs[:, None], ys[None, :])
BINS = [("0-20 m", 0, 20), ("20-40 m", 20, 40), ("40-60 m", 40, 61), ("all", 0, 61)]
K3 = np.ones((3, 3), np.uint8)
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))                      # radius 2 cells = 1 m


def shifted(m, cells=10):
    out = np.zeros_like(m)
    out[:, cells:] = m[:, :-cells]
    return out


acc = {k: np.zeros(8) for k in [b[0] for b in BINS]}          # static, hit, null hit, proxy cells, proxy with return, null proxy with return
cat_cnt = {"in a stationary box": 0, "within 1 m of the drivable edge (band)": 0, "inside the drivable area (> 1 m from the edge)": 0,
           "outside the drivable area (> 1 m from the edge)": 0}
dist_all = []
n_frames = 0
for i in range(0, len(gt.nusc.sample), step):
    s = gt.nusc.sample[i]
    g = gt.keyframe_gt(s["token"], grid)
    f = frames[i]
    m = (f["dt"] <= 0.5) & (f["vr"].abs() < 0.5)
    x, y = f["x"][m].numpy(), f["y"][m].numpy()
    ix, iy = np.floor((x - grid.x_min) / grid.res).astype(int), np.floor((y - grid.y_min) / grid.res).astype(int)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    ix, iy = ix[ok], iy[ok]
    drive = (g["drivable_area"] > 0.5).astype(np.uint8)
    inner = cv2.erode(drive, disk)
    outer = cv2.dilate(drive, disk)
    band = ((outer > 0) & (inner == 0)).astype(np.uint8)
    boxes_m = np.zeros_like(drive)
    for k, b in enumerate(g["meta"]["boxes"]):
        if b["observed"] and b["group"] in ("vehicle", "two_wheeler", "pedestrian") and (b["stationary"] or np.linalg.norm(b["vel"]) < 0.5):
            boxes_m[g["instance"] == k] = 1
    proxy = ((boxes_m > 0) | (band > 0)).astype(np.uint8)
    pd, nd = cv2.dilate(proxy, K3), cv2.dilate(shifted(proxy), K3)
    sret = np.zeros(drive.shape, np.uint8)
    np.add.at(sret, (ix, iy), 1)
    sd = cv2.dilate((sret > 0).astype(np.uint8), K3)
    dist_to_drive = cv2.distanceTransform((drive == 0).astype(np.uint8), cv2.DIST_L2, 5) * grid.res        # metres, 0 inside the area
    for name, lo, hi in BINS:
        r = (rng >= lo) & (rng < hi)
        sel = r[ix, iy]
        a = acc[name]
        a[0] += sel.sum()
        a[1] += pd[ix, iy][sel].sum()
        a[2] += nd[ix, iy][sel].sum()
        pc, nc = (proxy > 0) & r, (shifted(proxy) > 0) & r
        a[3] += pc.sum()
        a[4] += (sd > 0)[pc].sum()
        a[5] += nc.sum()
        a[6] += (sd > 0)[nc].sum()
    for cx, cy in zip(ix, iy):
        if boxes_m[cx, cy]:
            cat_cnt["in a stationary box"] += 1
        elif band[cx, cy]:
            cat_cnt["within 1 m of the drivable edge (band)"] += 1
        elif drive[cx, cy]:
            cat_cnt["inside the drivable area (> 1 m from the edge)"] += 1
        else:
            cat_cnt["outside the drivable area (> 1 m from the edge)"] += 1
    dist_all.append(dist_to_drive[ix, iy])
    n_frames += 1
print(f"{n_frames} keyframes of nuScenes mini; static returns = |ego-compensated radial velocity| < 0.5 m/s, 0.5 s window")
print(f"{'range':8s}{'static / frame':>15s}{'recall proxy':>14s}{'recall NULL':>13s}{'precision proxy':>17s}{'precision NULL':>16s}")
for name, _, _ in BINS:
    a = acc[name]
    print(f"{name:8s}{a[0] / n_frames:15.0f}{100 * a[1] / max(a[0], 1):13.1f}%{100 * a[2] / max(a[0], 1):12.1f}%"
          f"{100 * a[4] / max(a[3], 1):16.1f}%{100 * a[6] / max(a[5], 1):15.1f}%")
tot = sum(cat_cnt.values())
print("where the static returns are: " + "; ".join(f"{k} {100 * v / tot:.1f} %" for k, v in cat_cnt.items()))
d = np.concatenate(dist_all)
print("distance of static returns from the drivable area (m): median {:.1f}, share within 1 / 2 / 5 / 10 m: {}".format(
    np.median(d), " / ".join(f"{100 * (d <= t).mean():.0f} %" for t in (1, 2, 5, 10))))

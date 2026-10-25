"""Gate test of lift PD on the synthetic world (plan.md 10.1): camera image + EXACT depth -> lift -> right BEV cells?

Features per pixel: class one-hot (road, lane line, vehicle, pedestrian); depth: the exact depth along the optical
axis as a two-bin linear distribution over the depth bins of the camera head (1.25 + 0.5 i m, 118 bins), so a perfect
lift has all the information. Known answer: the 3D hit point of every pixel (z_cam K^-1 [u, v, 1], into the car frame)
-> the BEV cells it falls in, per class. Measured per class on random scenes: recall (hit cells where the lift put
mass) and precision (lifted cells that are hit cells, 1-cell tolerance), per range band.
Variants: height slices of training/lift_pd.Grid (-0.75 .. 2.75 m), slices with one at z = 0, and vertical
oversampling (4 sub-heights per 0.5 m slice, averaged). Feature resolution: full image (stride 1) and stride 8.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_oracle_lift.py [out.png]
"""
import math
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid, build_table, lift_pd  # noqa: E402
from synthetic_world import camera, front_camera, random_scene  # noqa: E402

DC = 1.25 + 0.5 * torch.arange(118, dtype=torch.float32)          # the camera head's depth bin centres
NAMES = ["road", "lane line", "vehicle", "pedestrian"]


def features(scene, cls, stride):
    """class one-hot (4, H, W), pooled to the stride."""
    ped = torch.zeros_like(cls, dtype=torch.bool)
    veh = torch.zeros_like(ped)
    for k, b in enumerate(scene.boxes):
        (ped if b.cls == "pedestrian" else veh).__ior__(cls == 3 + k)
    f = torch.stack([cls == 1, cls == 2, veh, ped]).float()
    return F.avg_pool2d(f[None], stride)[0] if stride > 1 else f


def depth_hist(depth, stride):
    """soft depth distribution (D, h, w): the normalised two-bin histogram of ALL pixel depths of a stride cell."""
    d = depth.clamp(DC[0].item(), DC[-1].item())
    idx = (d - DC[0]) / 0.5
    i0 = idx.floor().long().clamp(0, len(DC) - 2)
    a = (idx - i0).float()
    ok = torch.isfinite(depth).float()
    p = torch.zeros(len(DC), *d.shape)
    p.scatter_(0, i0[None], ((1 - a) * ok)[None])
    p.scatter_add_(0, (i0 + 1)[None], (a * ok)[None])
    p = F.avg_pool2d(p[None], stride)[0]
    return p / p.sum(0, keepdim=True).clamp(min=1e-6)


def depth_dist(depth, stride, mode="min"):
    """two-bin linear distribution (D, H, W) of the exact depth (min depth per stride cell, like the depth GT
    recipe of BEVDepth / DualBEV); mode "hist": all depths of the cell (depth_hist)."""
    if mode == "hist" and stride > 1:
        return depth_hist(depth, stride)
    if stride > 1:
        d = -F.max_pool2d(-depth[None, None], stride)[0, 0]
    else:
        d = depth
    d = d.clamp(DC[0].item(), DC[-1].item())
    idx = ((d - DC[0]) / 0.5)
    i0 = idx.floor().long().clamp(0, len(DC) - 2)
    a = idx - i0
    p = torch.zeros(len(DC), *d.shape)
    p.scatter_(0, i0[None], (1 - a)[None])
    p.scatter_add_(0, (i0 + 1)[None], a[None])
    p[:, ~torch.isfinite(depth if stride == 1 else d)] = 0
    return p


def hit_cells(depth, cls, scene, K, T, grid):
    """known answer: per class the set of BEV cells containing 3D hit points with -1 m <= z <= 3 m."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    ok = torch.isfinite(depth)
    z = depth.double()
    pc = torch.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, torch.ones_like(z)], -1)[ok]
    pw = pc @ torch.linalg.inv(T).T
    c = cls[ok]
    ix = ((pw[:, 0] - grid.x_min) / grid.res).floor().long()
    iy = ((pw[:, 1] - grid.y_min) / grid.res).floor().long()
    inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny) & (pw[:, 2] >= -1) & (pw[:, 2] <= 3)
    ped = torch.zeros_like(c, dtype=torch.bool)
    veh = torch.zeros_like(ped)
    for k, b in enumerate(scene.boxes):
        (ped if b.cls == "pedestrian" else veh).__ior__(c == 3 + k)
    masks = [c == 1, c == 2, veh, ped]
    out = torch.zeros(4, grid.nx, grid.ny, dtype=torch.bool)
    for j, m in enumerate(masks):
        sel = inside & m
        out[j, ix[sel], iy[sel]] = True
    return out


def lifted(feat, prob, K, T, hw, grid, z_levels, sub=1):
    """lift PD; with sub > 1 every slice is the mean of `sub` sub-heights spread over its 0.5 m."""
    zs = torch.tensor(z_levels, dtype=torch.float32)
    if sub > 1:
        off = (torch.arange(sub, dtype=torch.float32) + 0.5) / sub * 0.5 - 0.25
        zs = (zs.view(-1, 1) + off.view(1, -1)).reshape(-1)
    tab = build_table(grid, zs, K.float(), T.float(), hw, DC)
    vol = lift_pd(feat[None], prob[None], tab)[0]                   # (C, X, Y, Z*sub)
    if sub > 1:
        vol = vol.view(vol.shape[0], vol.shape[1], vol.shape[2], -1, sub).mean(-1)
    return vol.amax(-1)                                              # strongest height per cell


def score(lift, hits, rel=0.05):
    rows = []
    for j in range(4):
        m = lift[j] > rel * max(lift[j].max().item(), 1e-9)
        h = hits[j]
        hd = F.max_pool2d(h[None, None].float(), 3, 1, 1)[0, 0] > 0
        md = F.max_pool2d(m[None, None].float(), 3, 1, 1)[0, 0] > 0
        rec = (h & md).sum().item() / max(1, h.sum().item())
        prec = (m & hd).sum().item() / max(1, m.sum().item())
        rows.append((rec, prec, h.sum().item()))
    return rows


def range_recall(lift, hits, j, bands=((0, 20), (20, 40), (40, 60)), grid=Grid(), rel=0.05):
    xs, _ = grid.xy_centres()
    m = lift[j] > rel * max(lift[j].max().item(), 1e-9)
    md = F.max_pool2d(m[None, None].float(), 3, 1, 1)[0, 0] > 0
    out = []
    for lo, hi in bands:
        band = ((xs >= lo) & (xs < hi)).view(-1, 1)
        h = hits[j] & band
        out.append((h & md).sum().item() / max(1, h.sum().item()))
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    grid = Grid()
    K, T, hw = front_camera()
    rng = random.Random(7)
    Z0 = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    variants = {"slices -0.75..2.75 (as Grid)": (grid.z_levels, 1, "min"),
                "slices incl. z = 0, min depth": (Z0, 1, "min"),
                "slices incl. z = 0, depth histogram": (Z0, 1, "hist"),
                "oversampling x4, depth histogram": (grid.z_levels, 4, "hist"),
                "z = 0 slices + oversampling x4 + histogram": (Z0, 4, "hist")}
    acc = {(v, s): [] for v in variants for s in (1, 8)}
    lanes_by_range = {(v, s): [] for v in variants for s in (1, 8)}
    for n in range(6):
        sc = random_scene(rng)
        img, depth, cls = camera(sc, K, T, hw)
        hits = hit_cells(depth, cls, sc, K, T, grid)
        for stride in (1, 8):
            feat = features(sc, cls, stride)
            for name, (zl, sub, mode) in variants.items():
                prob = depth_dist(depth, stride, mode)
                lf = lifted(feat, prob, K, T, hw, grid, zl, sub)
                acc[(name, stride)].append(score(lf, hits))
                lanes_by_range[(name, stride)].append(range_recall(lf, hits, 1))
    print("oracle lift on 6 synthetic scenes | variant | stride | recall / precision per class: "
          + " | ".join(NAMES) + " || lane-line recall at 0-20 / 20-40 / 40-60 m")
    for (name, stride), runs in acc.items():
        cols = []
        for j in range(4):
            rec = sum(r[j][0] for r in runs) / len(runs)
            prec = sum(r[j][1] for r in runs) / len(runs)
            cols.append(f"{rec:.2f} / {prec:.2f}")
        lr = lanes_by_range[(name, stride)]
        bands = " / ".join(f"{sum(x[i] for x in lr) / len(lr):.2f}" for i in range(3))
        print(f"{name:30s} | {stride} | " + " | ".join(cols) + f" || {bands}")
    # gate (plan 10.1): the recommended lift (oversampling x4 + histogram depth) at stride 8
    key = ("oversampling x4, depth histogram", 8)
    runs, lr = acc[key], lanes_by_range[key]
    lane_rec = sum(r[1][0] for r in runs) / len(runs)
    veh_rec = sum(r[2][0] for r in runs) / len(runs)
    far = sum(x[2] for x in lr) / len(lr)
    ok = lane_rec >= 0.9 and veh_rec >= 0.9 and far >= 0.7
    print(f"GATE oracle lift ({key[0]}, stride 8): lanes {lane_rec:.2f} (>= 0.9), vehicles {veh_rec:.2f} (>= 0.9), "
          f"lanes 40-60 m {far:.2f} (>= 0.7) -> {'PASS' if ok else 'FAIL'}")
    if len(sys.argv) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc = random_scene(random.Random(7))
        img, depth, cls = camera(sc, K, T, hw)
        hits = hit_cells(depth, cls, sc, K, T, grid)
        fig, ax = plt.subplots(1, 5, figsize=(22, 4.4), gridspec_kw={"width_ratios": [2.75, 1, 1, 1, 1]})
        ax[0].imshow(img.permute(1, 2, 0).numpy())
        ax[0].set_title("synthetic camera image")
        show = [("known answer: hit cells (road, lanes, vehicles)", (hits[0].float() * 0.3 + hits[1].float() + hits[2].float() * 0.6))]
        for name, (zl, sub, mode) in list(variants.items())[1:]:
            lf = lifted(features(sc, cls, 8), depth_dist(depth, 8, mode), K, T, hw, grid, zl, sub)
            show.append((name + ", stride 8", (lf[0] / lf[0].max()) * 0.3 + lf[1] / lf[1].max() + (lf[2] / lf[2].max()) * 0.6))
        for a, (t, m) in zip(ax[1:], show):
            a.imshow(m.flip(0).flip(1).numpy(), cmap="magma", vmin=0, vmax=1, extent=[26, -26, 0, 60])
            a.set_title(t, fontsize=8)
        fig.tight_layout()
        fig.savefig(sys.argv[1], dpi=90)
        print("saved", sys.argv[1])
    print(f"GATE oracle lift -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

"""Is the shift of the projected GT lines quantisation, rounding, or an error? (Mirko, 2026-10-25)

Two tests on nuScenes mini keyframes:
1. picture (scene-0103, the Boston frame): the exact map polylines (vectors, sampled every 0.05 m, no grid) and the
   GT cell centres at 0.5 m projected into the front image side by side, zoomed on the road
2. numbers: the lateral brightness-contrast profile (paint minus 0.4 m to either side) for the VECTOR points and for
   the CELL centres (0.5 m), dividers at 6-20 m, lateral shifts -1.5..+1.5 m; if the vectors peak at 0 and the cells
   do not, the shift is quantisation; if both peak off 0, it is an error in the map, the calibration or the pose
Pixel rounding is excluded by bilinear sampling of the image (no integer truncation).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_gt_vector_vs_cells.py <png>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT, cell_centres  # noqa: E402
from lift_pd import Grid  # noqa: E402
from view_training_samples import project  # noqa: E402

G = Grid()
SHIFTS = np.arange(-1.5, 1.51, 0.25)
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")
gx, gy = cell_centres(G)


def dense(polylines, step=0.05):
    out = []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 2))


def bilinear(img, uv):
    H, W = img.shape
    u, v = uv[:, 0], uv[:, 1]
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u0, v0 = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    fu, fv = u[ok] - u0, v[ok] - v0
    val = np.full(len(u), np.nan)
    val[ok] = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
               + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
    return val


def profile(gt, img, xy):
    prof = []
    for dy in SHIFTS:
        def b(d):
            uv, z = project(gt, np.c_[xy[:, 0], xy[:, 1] + d, np.zeros(len(xy))])
            v = bilinear(img, uv)
            v[z <= 1] = np.nan
            return v
        c = b(dy) - 0.5 * (b(dy - 0.4) + b(dy + 0.4))
        prof.append(np.nanmean(c) if np.isfinite(c).sum() > 10 else np.nan)
    p = np.array(prof)
    return p - np.nanmedian(p)


res = {"vectors": [], "cells 0.5 m": []}
boston = None
for sc in src.nusc.scene:
    tok = sc["first_sample_token"]
    for k in range(40):
        s = src.nusc.get("sample", tok)
        if k % 8 == 4:
            gt = src.keyframe_gt(tok, G)
            img = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("L"), np.float32)
            vec = dense(gt["meta"]["vectors"]["lane_divider"] + gt["meta"]["vectors"]["road_divider"])
            vec = vec[(vec[:, 0] > 6) & (vec[:, 0] < 20)] if len(vec) else vec
            m = ((gt["lane_divider"] > 0.5) | (gt["road_divider"] > 0.5)) & (gx > 6) & (gx < 20)
            cells = np.stack([gx[m], gy[m]], 1)
            if len(vec) > 50 and len(cells) > 20:
                pv, pc = profile(gt, img, vec), profile(gt, img, cells)
                if np.isfinite(pv).all() and np.isfinite(pc).all():
                    res["vectors"].append(pv)
                    res["cells 0.5 m"].append(pc)
            if sc["name"] == "scene-0103" and k == 12:
                boston = (gt, img, vec, cells)
        if not s["next"]:
            break
        tok = s["next"]

print("shift (m):     " + " ".join(f"{d:+5.2f}" for d in SHIFTS))
for name, profs in res.items():
    P = np.array(profs)
    peaks = SHIFTS[np.argmax(P, 1)]
    mean = P.mean(0)
    print(f"{name:13s}: " + " ".join(f"{v:+5.1f}" for v in mean)
          + f" | frames {len(P)}, peak of mean {SHIFTS[int(np.argmax(mean))]:+.2f} m, per-frame median "
          f"{np.median(peaks):+.2f} m, within +-0.25 m {np.mean(np.abs(peaks) <= 0.25):.0%}")

gt, img, vec, cells = boston
fig, ax = plt.subplots(1, 2, figsize=(16, 5.4))
rgb = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("RGB"))
for a, (pts, title, col) in zip(ax, [(vec, "exact map polylines (vectors), no grid", (1, 0.2, 0.2)),
                                    (cells, "GT cell centres, 0.5 m grid", (0.2, 0.6, 1))]):
    a.imshow(rgb)
    uv, z = project(gt, np.c_[pts, np.zeros(len(pts))])
    ok = z > 1
    a.scatter(uv[ok, 0], uv[ok, 1], s=4 if pts is vec else 14, color=col, linewidths=0)
    a.set_xlim(0, 1600)
    a.set_ylim(900, 450)
    a.set_title(f"scene-0103: {title}", fontsize=10)
    a.axis("off")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
print("saved", sys.argv[1])

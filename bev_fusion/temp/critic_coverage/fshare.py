"""footprint_share vs the exact area share (shapely) on real boxes + real fusion masks, and on random rotated boxes."""
import os, sys, math
import numpy as np
import shapely
from shapely.geometry import Polygon, box as shbox
from shapely.ops import unary_union
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec

grid = GridSpec()
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(5)


def exact_share(fp, mask):
    poly = Polygon(fp)
    if poly.area <= 0:
        return float("nan")
    x0, y0, x1, y1 = poly.bounds
    i0 = max(int(math.floor((x0 - grid.x_min) / grid.res)), 0); i1 = min(int(math.floor((x1 - grid.x_min) / grid.res)), grid.nx - 1)
    j0 = max(int(math.floor((y0 - grid.y_min) / grid.res)), 0); j1 = min(int(math.floor((y1 - grid.y_min) / grid.res)), grid.ny - 1)
    area = 0.0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if mask[i, j]:
                cell = shbox(grid.x_min + i * grid.res, grid.y_min + j * grid.res, grid.x_min + (i + 1) * grid.res, grid.y_min + (j + 1) * grid.res)
                area += poly.intersection(cell).area
    return area / poly.area


res = {}
pick = rng.choice(len(files), 120, replace=False)
for i in pick:
    g = load_keyframe(d, files[i][:-4])
    fus = C.fusion_cells(g)
    for b in g["meta"]["boxes"]:
        fp = b["corners"]
        P = Polygon(fp)
        if not P.is_valid or P.area < 1e-3:
            res.setdefault("invalid", []).append(1); continue
        f = C.footprint_share(fp, fus, grid)
        e = exact_share(fp, fus)
        # only informative boxes: not all 0 / 1 in a trivial way
        res.setdefault(b["group"], []).append((abs(f - e), P.area, f, e))
for k, v in res.items():
    if k == "invalid":
        print("invalid footprints:", len(v)); continue
    err = np.array([x[0] for x in v]); nontriv = np.array([0.001 < x[3] < 0.999 for x in v])
    print(f"{k:12s} n={len(v):5d}  nontrivial (0<e<1)={nontriv.sum():4d}  max err {err.max():.3f}  p99 {np.quantile(err, .99):.3f}  | nontrivial: max {err[nontriv].max() if nontriv.any() else 0:.3f} mean {err[nontriv].mean() if nontriv.any() else 0:.3f} p95 {np.quantile(err[nontriv], .95) if nontriv.any() else 0:.3f}")

# random rotated boxes on a random-boundary mask (half-plane at random angle, so that the share is spread between 0 and 1) and a real mask
print("random rotated boxes against a half-plane mask (the mask edge crosses the box):")
for name, (l, w) in {"car 4.5x1.9": (4.5, 1.9), "ped 0.6x0.6": (0.6, 0.6), "bike 1.8x0.7": (1.8, 0.7), "truck 12x2.5": (12, 2.5), "ped small 0.4x0.5": (0.5, 0.4)}.items():
    errs = []
    for k in range(400):
        cx, cy = rng.uniform(10, 50), rng.uniform(-20, 20); yaw = rng.uniform(-math.pi, math.pi)
        loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        c, s = math.cos(yaw), math.sin(yaw)
        fp = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
        gx, gy = grid.centres()
        th = rng.uniform(0, 2 * math.pi); off = rng.uniform(-0.3, 0.3) * max(l, w)
        mask = ((gx - cx) * math.cos(th) + (gy - cy) * math.sin(th)) > off
        f = C.footprint_share(fp, mask, grid); e = exact_share(fp, mask)
        errs.append(abs(f - e))
    errs = np.array(errs)
    print(f"  {name:18s} max {errs.max():.3f} p95 {np.quantile(errs, .95):.3f} mean {errs.mean():.4f}")
print("unit checks: footprint at the grid edge")
fp = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]]); print("  straddles x_min:", C.footprint_share(fp, np.ones((grid.nx, grid.ny), bool), grid), "exact 0.5")
fp = np.array([[59.0, 25], [61, 25], [61, 27], [59, 27]]); print("  corner outside both x_max and y_max:", C.footprint_share(fp, np.ones((grid.nx, grid.ny), bool), grid), "exact 0.25")

import math, numpy as np
from shapely.geometry import Polygon, box as shbox
from autogt import coverage as C
from autogt.grid import GridSpec
grid = GridSpec(); rng = np.random.default_rng(6)
def exact(fp, mask):
    poly = Polygon(fp); x0, y0, x1, y1 = poly.bounds; a = 0.0
    for i in range(max(int((x0 - grid.x_min) // grid.res), 0), min(int((x1 - grid.x_min) // grid.res), grid.nx - 1) + 1):
        for j in range(max(int((y0 - grid.y_min) // grid.res), 0), min(int((y1 - grid.y_min) // grid.res), grid.ny - 1) + 1):
            if mask[i, j]:
                a += poly.intersection(shbox(grid.x_min + i * grid.res, grid.y_min + j * grid.res, grid.x_min + (i + 1) * grid.res, grid.y_min + (j + 1) * grid.res)).area
    return a / poly.area
gx, gy = grid.centres()
for name, (l, w) in {"ped 0.6x0.6": (0.6, 0.6), "ped 0.4x0.5": (0.5, 0.4)}.items():
    cases = []
    for k in range(300):
        cx, cy = rng.uniform(10, 50), rng.uniform(-20, 20); yaw = rng.uniform(-math.pi, math.pi)
        loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]); c, s = math.cos(yaw), math.sin(yaw)
        fp = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
        th = rng.uniform(0, 2 * math.pi); off = rng.uniform(-0.3, 0.3) * max(l, w)
        mask = ((gx - cx) * math.cos(th) + (gy - cy) * math.sin(th)) > off
        cases.append((fp, mask, exact(fp, mask)))
    for step in (0.125, 0.0625, 0.03125):
        e = np.array([abs(C.footprint_share(fp, m, grid, step=step) - ex) for fp, m, ex in cases])
        print(name, "step", step, "max %.3f p95 %.3f mean %.4f" % (e.max(), np.quantile(e, .95), e.mean()))

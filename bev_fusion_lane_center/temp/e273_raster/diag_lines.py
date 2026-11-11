"""scratch: do the dividers / boundary lie on the drivable area; missing areas"""
import os, sys
import numpy as np, shapely
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(200)
tot = {k: 0 for k in QR.LINES}; ins = {k: 0 for k in QR.LINES}; near = {k: 0 for k in QR.LINES}
no_road = no_inter = pieces_no_road = no_lines = 0
small = 0
for g in fr:
    road = QR._polygons(g["areas"]["road"], 0.0)
    no_road += not road
    no_inter += (not g["areas"]["intersection"]) and any(e["type"] in ("stop", "exit") for e in g["events"])
    if g["pieces"] and not road: pieces_no_road += 1
    if not any(len(g["lines"][k]) for k in QR.LINES): no_lines += 1
    if not road: continue
    ru = shapely.union_all(road); shapely.prepare(ru)
    rb = ru.boundary; shapely.prepare(rb)
    for k in QR.LINES:
        polys = QR._polylines(g["lines"][k])
        if not polys: continue
        p = QR._samples(polys, 0.5)
        p = p[(np.abs(p) < 49.5).all(1)]
        tot[k] += len(p); ins[k] += int(shapely.contains_xy(ru, p[:, 0], p[:, 1]).sum())
        near[k] += int((shapely.distance(rb, shapely.points(p)) <= 0.3).sum())
print("frames", len(fr), "without road polygon:", no_road, "; with stop/exit events but no intersection polygon:", no_inter, "; no lines at all:", no_lines)
for k in QR.LINES:
    print(f"{k}: samples {tot[k]}, inside the road polygon {ins[k] / max(tot[k], 1):.3f}, within 0.3 m of its outline {near[k] / max(tot[k], 1):.3f}")

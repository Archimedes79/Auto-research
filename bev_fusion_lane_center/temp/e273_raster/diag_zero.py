"""scratch: connector segments flagged in_intersection that lie in no intersection polygon"""
import os, sys
import numpy as np, shapely
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(120)
rows = []
for k, g in enumerate(fr):
    polys = QR._polygons(g["areas"]["intersection"], 0.0)
    un = shapely.union_all(polys) if polys else None
    if un is not None: shapely.prepare(un)
    road = QR._polygons(g["areas"]["road"], 0.0)
    ru = shapely.union_all(road) if road else None
    if ru is not None: shapely.prepare(ru)
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        for s in p["segments"]:
            if not s["in_intersection"]: continue
            pts = q[s["i0"]: s["i1"] + 1]
            ins = shapely.contains_xy(un, pts[:, 0], pts[:, 1]) if un is not None else np.zeros(len(pts), bool)
            rin = shapely.contains_xy(ru, pts[:, 0], pts[:, 1]).mean() if ru is not None else 0
            rows.append((k, ins.mean(), 50 - np.abs(pts).max(1).max(), 50 - np.abs(pts).max(1).min(), len(pts), rin, len(polys), p["kind"], p["road"]))
R = rows
z = [r for r in R if r[1] == 0]
print("in_intersection segments", len(R), "with 0 inside:", len(z), " frames with no intersection polygon at all:", len({r[0] for r in z if r[6] == 0}))
print("zero-inside: n points pct 10/50/90:", np.percentile([r[4] for r in z], [10, 50, 90]), "; closest approach to the window border pct 10/50/90:", np.percentile([r[2] for r in z], [10, 50, 90]), "; share of points inside the road polygon:", np.mean([r[5] for r in z]))
print("zero-inside with no polygon in the frame:", sum(1 for r in z if r[6] == 0), "; in a frame with polygons:", sum(1 for r in z if r[6] > 0))
low = [r for r in R if 0 < r[1] < 0.5]
print("0 < share < 0.5:", len(low), " (window cuts a connector: the 50 % rule was applied to the whole connector, not the part in the window)")

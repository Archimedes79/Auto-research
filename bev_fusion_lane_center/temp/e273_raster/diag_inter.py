"""scratch: do the intersection polygons cover the connector segments the GT calls in_intersection"""
import os, sys
import numpy as np, shapely
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(120)
tot = inn = cellin = 0
lane_tot = lane_in = 0
fr_inside = []
dist_out = []
conn_out_tot = conn_out_in = 0
for g in fr:
    polys = QR._polygons(g["areas"]["intersection"], 0.0)
    r2 = QR.areas2(g)
    un = shapely.union_all(polys) if polys else None
    shapely.prepare(un) if un is not None else None
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        for s in p["segments"]:
            pts = q[s["i0"]: s["i1"] + 1]
            if not len(pts): continue
            inside = shapely.contains_xy(un, pts[:, 0], pts[:, 1]) if un is not None else np.zeros(len(pts), bool)
            c = np.clip(np.floor((pts + 50) / 2).astype(int), 0, 49)
            cin = r2[1][c[:, 0], c[:, 1]] > 0
            if s["in_intersection"]:
                tot += len(pts); inn += int(inside.sum()); cellin += int(cin.sum())
                fr_inside.append(inside.mean())
                if un is not None:
                    dist_out += list(shapely.distance(un, shapely.points(pts[~inside])))
            elif s["kind"] == "connector":
                conn_out_tot += len(pts); conn_out_in += int(inside.sum())
            else:
                lane_tot += len(pts); lane_in += int(inside.sum())
fi = np.array(fr_inside)
print(f"points of in_intersection segments: {tot}; inside a polygon {inn / tot:.3f}; in an intersection cell {cellin / tot:.3f}")
print("per segment share inside: pct 5/25/50/75:", np.percentile(fi, [5, 25, 50, 75]), " segments with <50% inside:", (fi < .5).mean(), " with 0 inside:", (fi == 0).mean())
d = np.array(dist_out)
print("outside points: distance to polygons pct 50/90/99/max:", np.percentile(d, [50, 90, 99, 100]) if len(d) else None)
print(f"connector segments NOT in_intersection: points {conn_out_tot}, inside a polygon {conn_out_in / max(conn_out_tot, 1):.3f};  lane points {lane_tot}, inside a polygon {lane_in / lane_tot:.3f}")

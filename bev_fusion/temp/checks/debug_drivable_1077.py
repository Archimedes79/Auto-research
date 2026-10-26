"""scene-1077 keyframe 8: the BEV picture shows no drivable area under the road. Is it empty in the GT, or the picture?"""
import os
import sys

import numpy as np

sys.path.insert(0, "training")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402

G = Grid()
src = NuScenesGT(os.path.expanduser("~/data/nuscenes"), "v1.0-mini")
sc = next(s for s in src.nusc.scene if s["name"] == "scene-1077")
tok = sc["first_sample_token"]
for _ in range(8):
    tok = src.nusc.get("sample", tok)["next"]
gt = src.keyframe_gt(tok, G)
da = gt["drivable_area"]
print("log location:", src.nusc.get("log", sc["log_token"])["location"])
print("drivable coverage mean", da.mean(), " cells > 0.5:", int((da > 0.5).sum()))
for x in (3, 10, 20, 30, 50):
    row = da[int(x / G.res)]
    ys = G.y_min + (np.nonzero(row > 0.5)[0] + 0.5) * G.res
    print(f"x={x:2d} m: drivable y ranges:", (ys.min(), ys.max(), len(ys)) if len(ys) else "none")
nmap = src.map_for(sc["token"])
T = gt["meta"]["T_e2g"]
c = (T @ np.array([20, 0, 0, 1]))[:2]
recs = nmap.get_records_in_patch((c[0] - 30, c[1] - 30, c[0] + 30, c[1] + 30), ["drivable_area", "road_segment", "lane"], mode="intersect")
print({k: len(v) for k, v in recs.items()})
Tg2e = np.linalg.inv(T)
for t in recs["drivable_area"]:
    for p in nmap.get("drivable_area", t)["polygon_tokens"]:
        poly = nmap.extract_polygon(p)
        xy = np.asarray(poly.exterior.coords)[:, :2]
        e = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ Tg2e.T)[:, :2]
        print("polygon area", round(poly.area), "ego-frame x", e[:, 0].min().round(), e[:, 0].max().round(), "y", e[:, 1].min().round(), e[:, 1].max().round(),
              "holes", len(poly.interiors))

from shapely.geometry import Point
pe = (T @ np.array([0, 0, 0, 1]))[:2]
print("ego global", pe)
for t in recs["drivable_area"]:
    toks = nmap.get("drivable_area", t)["polygon_tokens"]
    print("record", t[:8], "polygon tokens", [(p or "None")[:8] for p in toks])
    for p in [q for q in toks if q]:
        poly = nmap.extract_polygon(p)
        print("  valid", poly.is_valid, "contains ego", poly.contains(Point(pe)), "contains 20m ahead", poly.contains(Point((T @ np.array([20, 0, 0, 1]))[:2])),
              "in a hole:", any(__import__('shapely').geometry.Polygon(h).contains(Point(pe)) for h in poly.interiors))

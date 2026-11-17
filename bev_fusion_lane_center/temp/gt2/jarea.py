import sys, numpy as np
sys.path.insert(0, "../../tools/viz"); sys.path.insert(0, "../../training/grid")
import qgraph_view as QV
from shapely.geometry import Polygon
from shapely.ops import unary_union
toks = []
for k, (n, tk) in QV.val_scenes().items():
    toks += tk[::6]
tot_id = tot_none = road = 0.0; fr_none = 0; shares = []
for t in toks:
    g = QV.load_gt(t)
    ids = [Polygon(a["exterior"]).buffer(0) for a in g["areas"]["intersection"] if a.get("id") is not None and len(a["exterior"]) > 2]
    non = [Polygon(a["exterior"]).buffer(0) for a in g["areas"]["intersection"] if a.get("id") is None and len(a["exterior"]) > 2]
    ai = sum(p.area for p in ids); an = sum(p.area for p in non)
    tot_id += ai; tot_none += an; fr_none += int(an > 0)
    rd = unary_union([Polygon(a["exterior"]).buffer(0) for a in g["areas"]["road"] if len(a["exterior"]) > 2]).area if g["areas"]["road"] else 0
    road += rd
    if ai + an > 0 and rd > 0: shares.append((ai + an) / rd)
print("frames", len(toks), "area with connectors", round(tot_id), "area without connectors (id None)", round(tot_none), "share none", round(tot_none / (tot_id + tot_none), 3), "frames with none-polygons", round(fr_none / len(toks), 3))
print("intersection share of road area per frame (median, p90):", np.round(np.percentile(shares, [50, 90]), 3))

"""E276 pre-experiment P7 (video note 1: "the intersection label did not always look good"), CPU: how consistent is the GT intersection label with the lane GT itself? On the 873 VAL frames of the l4 store:
(a) the share of the connector points that run inside an intersection (segments flagged in_intersection) which lie inside the intersection polygons (0 m, 1 m, 2 m dilation, the head's target uses 2 m);
(b) the share of the polygon area (rasterised 1 m) that has no lane point within 6 m (an intersection polygon where no lane runs: an empty label);
(c) the distance of the stop / exit event points to the nearest polygon (the card of E273: they lie 1 - 3 m before the edge);
(d) frames with connectors in an intersection but no polygon at all, and frames with polygons but no connector inside.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_intersection_label.py <l4 store folder>
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "training"))
import qgraph_eval as QE  # noqa: E402

store = sys.argv[1]
frames = QE.val_frames()
cnt = dict(frames=0, conn_pts=0, in0=0, in1=0, in2=0, no_poly_with_conn=0, poly_without_conn=0, frames_with_poly=0, empty_area=0.0, poly_area=0.0, none_id_polys=0, polys=0)
d_stop = []
for f in frames:
    g = pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))
    cnt["frames"] += 1
    polys = []
    for a in g["areas"]["intersection"]:
        ext = np.asarray(a["exterior"], float)
        if len(ext) >= 3:
            p = Polygon(ext, [np.asarray(h, float) for h in a.get("holes", []) if len(h) >= 3])
            if p.is_valid and not p.is_empty:
                polys.append((p, a.get("id")))
    cnt["polys"] += len(polys)
    cnt["none_id_polys"] += sum(1 for _, i in polys if i is None)
    conn = []
    allpts = []
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        allpts.append(q)
        for s in p.get("segments", []):
            if s.get("kind") == "connector" and s.get("in_intersection"):
                conn.append(q[s["i0"]: s["i1"] + 1])
    conn = np.concatenate(conn) if conn else np.zeros((0, 2))
    union = unary_union([p for p, _ in polys]) if polys else None
    if len(conn):
        cnt["conn_pts"] += len(conn)
        if union is None:
            cnt["no_poly_with_conn"] += 1
        else:
            cnt["in0"] += int(shapely.contains_xy(union, conn[:, 0], conn[:, 1]).sum())
            cnt["in1"] += int(shapely.contains_xy(union.buffer(1.0), conn[:, 0], conn[:, 1]).sum())
            cnt["in2"] += int(shapely.contains_xy(union.buffer(2.0), conn[:, 0], conn[:, 1]).sum())
    if union is not None:
        cnt["frames_with_poly"] += 1
        if not len(conn):
            cnt["poly_without_conn"] += 1
        xs, ys = np.meshgrid(np.arange(-49.5, 50, 1.0), np.arange(-49.5, 50, 1.0))
        m = shapely.contains_xy(union, xs.ravel(), ys.ravel())
        cnt["poly_area"] += int(m.sum())
        if m.any() and len(np.concatenate(allpts)):
            far = cKDTree(np.concatenate(allpts)).query(np.c_[xs.ravel()[m], ys.ravel()[m]])[0] > 6.0
            cnt["empty_area"] += int(far.sum())
        for e in g["events"]:
            if e["type"] in ("stop", "exit"):
                pt = shapely.points(float(e["pos"][0]), float(e["pos"][1]))
                d_stop.append(float(union.distance(pt)))
d = np.array(d_stop)
out = dict(frames=cnt["frames"], connector_points=cnt["conn_pts"], inside_polygon=cnt["in0"] / max(cnt["conn_pts"], 1), inside_1m=cnt["in1"] / max(cnt["conn_pts"], 1), inside_2m=cnt["in2"] / max(cnt["conn_pts"], 1),
           frames_connectors_without_any_polygon=cnt["no_poly_with_conn"], frames_with_polygon=cnt["frames_with_poly"], frames_polygon_without_connector=cnt["poly_without_conn"],
           polygon_area_without_lane_within_6m=cnt["empty_area"] / max(cnt["poly_area"], 1), polygons=cnt["polys"], polygons_without_node_id=cnt["none_id_polys"],
           stop_exit_distance_to_polygon_m=dict(n=len(d), median=float(np.median(d)), p90=float(np.percentile(d, 90)), inside=float((d == 0).mean())) if len(d) else None)
json.dump(out, open(os.path.join(ROOT, "temp", "e276", "intersection_label.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

import json
import sys
import time

import numpy as np

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
import qgraph_view as QV
import jgt as J
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree

step = int(sys.argv[1]) if len(sys.argv) > 1 else 3
toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk[::step]
print("frames", len(toks), flush=True)
t0 = time.time()
R = dict(frames=0, junction_frames=0, zone_area=[], nu_area=[], nu_excess=[], nu_in_zone=[], spread_in=[], spread_out=[], n_in=[], n_out=[], maneuvers=0, multi=0, mono=0, ident=0,
         ends_total=0, ends_noevent=0, n_ent=[], n_pairs=[], pairs_driven=0, pairs_all=0, k_equal=0, k_total=0)
for ti, tok in enumerate(toks):
    g = QV.load_gt(tok)
    R["frames"] += 1
    # natural ends without an event
    ev = np.array([e["pos"] for e in g["events"]]).reshape(-1, 2)
    kd = cKDTree(ev) if len(ev) else None
    for pc in g["pieces"]:
        p = np.asarray(pc["pts"])
        for flag, pt in ((pc["start_open"], p[0]), (pc["end_open"], p[-1])):
            if flag or len(p) < 3:
                continue
            R["ends_total"] += 1
            if kd is None or kd.query(pt)[0] > 1.0:
                R["ends_noevent"] += 1
    jg = J.junction_gt(g)
    if jg is None:
        continue
    R["junction_frames"] += 1
    nu = [Polygon(a["exterior"]) for a in g["areas"]["intersection"] if a.get("id") is not None and len(a["exterior"]) > 2]
    nu = [p.buffer(0) for p in nu]
    for z in jg["zones"]:
        zs = z["shp"]
        close = [p for p in nu if p.intersects(zs)]
        if not close:
            continue
        U = unary_union(close)
        R["zone_area"].append(zs.area)
        R["nu_area"].append(U.area)
        R["nu_in_zone"].append(U.intersection(zs.buffer(3.0)).area / max(U.area, 1e-6))
        R["nu_excess"].append(U.difference(zs.buffer(3.0)).area)
    for a in jg["arms"]:
        if a["n_in"]:
            R["spread_in"].append(a["spread_in"])
            R["n_in"].append(a["n_in"])
        if a["n_out"]:
            R["spread_out"].append(a["spread_out"])
            R["n_out"].append(a["n_out"])
    o = jg["order"]
    R["maneuvers"] += o["maneuvers"]; R["multi"] += o["multi"]; R["mono"] += o["monotone"]; R["ident"] += o["identity"]
    R["n_ent"].append(len(jg["ent"])); R["n_pairs"].append(len(jg["pairs"]))
    R["pairs_driven"] += int(jg["pairs"][:, 9].sum()); R["pairs_all"] += len(jg["pairs"])
    if ti % 50 == 0:
        print(ti, time.time() - t0, flush=True)


def q(x, ps=(10, 50, 90)):
    return [round(float(np.percentile(x, p)), 2) for p in ps] if len(x) else None


S = dict(frames=R["frames"], junction_frames=R["junction_frames"],
         zone_area_q=q(R["zone_area"]), nuscenes_polygon_area_q=q(R["nu_area"]), area_ratio_nu_over_zone_median=round(float(np.median(np.array(R["nu_area"]) / np.array(R["zone_area"]))), 2),
         nu_share_inside_zone_plus3m_q=q(R["nu_in_zone"]), nu_excess_area_q=q(R["nu_excess"]),
         arm_spread_in_q=q(R["spread_in"]), arm_spread_out_q=q(R["spread_out"]), arm_spread_in_gt3=round(float(np.mean(np.array(R["spread_in"]) > 3.0)), 3), arms_in=len(R["spread_in"]),
         lanes_per_arm_in_hist=np.bincount(R["n_in"]).tolist(), lanes_per_arm_out_hist=np.bincount(R["n_out"]).tolist(),
         maneuvers=R["maneuvers"], multi_pair_maneuvers=R["multi"], monotone=R["mono"], identity=R["ident"],
         ends_total=R["ends_total"], ends_without_event=R["ends_noevent"], ends_without_event_share=round(R["ends_noevent"] / max(R["ends_total"], 1), 3),
         entries_per_junction_frame_q=q(R["n_ent"]), pairs_per_junction_frame_q=q(R["n_pairs"]), pairs_driven_share=round(R["pairs_driven"] / max(R["pairs_all"], 1), 3))
print(json.dumps(S, indent=1))
json.dump(S, open("jstats_val.json", "w"), indent=1)

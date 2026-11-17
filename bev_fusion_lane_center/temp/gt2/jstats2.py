import json
import sys

import numpy as np

sys.path.insert(0, "../../tools/viz")
sys.path.insert(0, "../../training/grid")
sys.path.insert(0, "../../training/data/prep")
import qgraph_view as QV
import jgt as J
import junction_paint as JP
from shapely.geometry import Point, Polygon

step = int(sys.argv[1]) if len(sys.argv) > 1 else 3
toks = []
for k, (name, tk) in QV.val_scenes().items():
    toks += tk[::step]
R = dict(core_over_zone=[], d_ent_stop=[], d_ent_ped=[], d_ext_ped=[], n_ent=0, has_core=0, jf=0, frames=0, no_paint=0)
for tok in toks:
    g = QV.load_gt(tok)
    R["frames"] += 1
    jg = J.junction_gt(g)
    if jg is None:
        continue
    R["jf"] += 1
    za = sum(z["area"] for z in jg["zones"])
    if jg["core"] is not None:
        R["has_core"] += 1
        R["core_over_zone"].append(jg["core"].area / max(za, 1e-6))
    try:
        paint = JP.load(tok)
    except FileNotFoundError:
        R["no_paint"] += 1
        continue
    stops = [Polygon(s["poly"]).buffer(0) for s in paint["stop_line"] if len(s["poly"]) > 2]
    peds = [Polygon(p).buffer(0) for p in paint["ped_crossing"] if len(p) > 2]
    for e in jg["ent"]:
        pt = Point(e[0], e[1])
        R["n_ent"] += 1
        R["d_ent_stop"].append(min([s.distance(pt) for s in stops], default=99.0))
        R["d_ent_ped"].append(min([s.distance(pt) for s in peds], default=99.0))
    for e in jg["ext"]:
        pt = Point(e[0], e[1])
        R["d_ext_ped"].append(min([s.distance(pt) for s in peds], default=99.0))


def q(x, ps=(10, 50, 90)):
    return [round(float(np.percentile(x, p)), 2) for p in ps] if len(x) else None


S = dict(frames=R["frames"], junction_frames=R["jf"], frames_without_paint_file=R["no_paint"], frames_with_core=R["has_core"], core_over_zone_q=q(R["core_over_zone"]),
         entry_to_stop_line_m_q=q(R["d_ent_stop"]), entries_within_3m_of_stop_line=round(float(np.mean(np.array(R["d_ent_stop"]) <= 3.0)), 3), entries_within_6m_of_stop_line=round(float(np.mean(np.array(R["d_ent_stop"]) <= 6.0)), 3),
         entry_to_crossing_m_q=q(R["d_ent_ped"]), entries_within_4m_of_crossing=round(float(np.mean(np.array(R["d_ent_ped"]) <= 4.0)), 3),
         exit_to_crossing_m_q=q(R["d_ext_ped"]), exits_within_4m_of_crossing=round(float(np.mean(np.array(R["d_ext_ped"]) <= 4.0)), 3), entries=R["n_ent"])
print(json.dumps(S, indent=1))
json.dump(S, open("jstats2_val.json", "w"), indent=1)

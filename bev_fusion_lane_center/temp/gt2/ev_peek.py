import sys, numpy as np
sys.path.insert(0, "../../tools/viz"); sys.path.insert(0, "../../training/grid")
import qgraph_view as QV
n_all = 0; n_in = 0; n_part = 0; unit_bad = 0
import jgt as J
for sid, (name, toks) in QV.val_scenes().items():
    for t in toks[::2]:
        g = QV.load_gt(t); jg = J.junction_gt(g)
        from shapely.geometry import Point
        for e in g["events"]:
            if e["type"] in ("split", "merge"):
                n_all += 1; n_part += int(bool(e.get("partial")))
                if jg is not None and any(z["shp"].buffer(2.0).contains(Point(*e["pos"])) for z in jg["zones"]): n_in += 1
                d = np.asarray(e["dirs_in"] if e["type"] == "split" else e["dirs_out"], float)
                unit_bad += int(len(d) == 0 or abs(np.linalg.norm(d[0]) - 1) > 1e-3)
print("split+merge events", n_all, "inside a zone+2m", n_in, "partial", n_part, "non-unit or empty direction", unit_bad)

import os, tempfile, traceback
import numpy as np
import qgraph_view as QV
QV._CACHE["infos"], QV._CACHE["frames"] = {}, {}
g0 = dict(pieces=[], events=[], lines=dict(lane_divider=[], road_divider=[], boundary=[]), areas=dict(road=[], intersection=[]), location="synthetic", signature="x", half=50.0, frame="lane")
g1 = QV._synthetic(); g1["pieces"] = g1["pieces"][3:]; g1["events"] = []          # one single piece (D), no events
out = tempfile.mkdtemp()
for name, g in (("empty", g0), ("one piece", g1)):
    for kind in ("oracle", "degraded"):
        try:
            pred = QV.make_pred(g, kind, "t")
            QV.render_frame("synthetic", pred, os.path.join(out, f"{name}_{kind}.png"), cams=False, g=g)
            print(name, kind, "ok")
        except Exception as e:
            print(name, kind, "FAILED:", type(e).__name__, str(e)[:100]); traceback.print_exc(limit=-3)

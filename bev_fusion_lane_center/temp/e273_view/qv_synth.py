import os, sys
import qgraph_view as QV
R = os.environ["BFLC_ROOT"]
out = os.path.join(R, "temp", "e273_view")
g = QV._synthetic()
for kind in ("oracle", "degraded"):
    QV.render_frame("synthetic", QV.make_pred(g, kind, "selftest"), os.path.join(out, f"synth_{kind}.png"), cams=False, g=g)
print("done")

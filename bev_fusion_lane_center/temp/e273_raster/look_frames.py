"""scratch: look at a few real lane GT frames"""
import os, sys, time
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import shapely
import scipy
print("shapely", shapely.__version__, "scipy", scipy.__version__, "numpy", np.__version__)
from test_qtargets import frames
t = time.time()
fr = frames(5)
print("load", time.time() - t)
for g in fr:
    print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in g.items() if k in ("pieces", "events", "roads")}, g["half"], g["frame"])
    L = g["lines"]
    print(" lines", {k: [(len(l["pts"]), str(l["pts"].dtype)) for l in v][:4] + ["..."] for k, v in L.items()})
    print(" line lengths", {k: round(sum(float(np.linalg.norm(np.diff(l["pts"], axis=0), axis=1).sum()) for l in v), 0) for k, v in L.items()})
    A = g["areas"]
    for k, v in A.items():
        print(" area", k, [(len(p["exterior"]), len(p["holes"]), round(p["area"])) for p in v])
    print(" ev", {t: sum(1 for e in g["events"] if e["type"] == t) for t in ("split", "merge", "cross", "stop", "exit", "start", "end", "window")})
    p0 = g["pieces"][0]
    print(" piece0", p0["kind"], p0["pts"].dtype, p0["pts"].shape, p0["segments"][:2])
    dd = np.concatenate([np.linalg.norm(np.diff(p["pts"], axis=0), axis=1) for p in g["pieces"] if len(p["pts"]) > 1])
    print(" piece step min/max", dd.min(), dd.max(), "n zero", (dd < 1e-6).sum(), "pieces<2", sum(len(p["pts"]) < 2 for p in g["pieces"]))
    if A["road"]:
        p = A["road"][0]
        print(" ext type", type(p["exterior"]), np.asarray(p["exterior"]).shape, [np.asarray(h).shape for h in p["holes"]][:3], np.asarray(p["exterior"])[:2])

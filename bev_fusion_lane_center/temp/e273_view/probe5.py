import os, sys, time, json, numpy as np
R = os.environ["BFLC_ROOT"]; HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(R, "..")); sys.path.insert(0, os.path.join(R, "training", "grid"))
t0 = time.time()
from autogt import lane_gt
print("import autogt", round(time.time() - t0, 2))
import qformat as QF, qtargets as QT
toks = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
sp = json.load(open(f"{R}/data/splits.json"))
rows = sp["val"]
pz = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")
sc = pz["scene"]
ix = [r for r in rows if sc[r] == 11]
ix = sorted(ix, key=lambda r: pz["ts"][r])
tok = toks[ix[12]]
t0 = time.time(); g = lane_gt.load(tok); print("load", round(time.time() - t0, 3), tok)
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in g.items()})
t0 = time.time(); pr = QF.oracle_prediction(g); print("oracle", round(time.time() - t0, 2))
print({k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in pr["nodes"].items()})
print({k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in pr["links"].items()})
print("long", pr["links"]["long"].sum(), "events", [e["type"] for e in g["events"]])
print("lines", {k: len(v) for k, v in g["lines"].items()}, "areas", {k: len(v) for k, v in g["areas"].items()})
p0 = g["pieces"][0]; print({k: (v if not hasattr(v, "shape") else v.shape) for k, v in p0.items()})
print(g["lines"]["boundary"][0].keys() if g["lines"]["boundary"] else None)
print(g["areas"]["road"][0].keys() if g["areas"]["road"] else None, g["areas"]["road"][0]["exterior"].shape)
import cv2, matplotlib; print(cv2.__version__, matplotlib.__version__)
print("scene", g["location"], g["sample"] == tok)

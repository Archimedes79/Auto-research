import json, os, sys, pickle, numpy as np, time
import bflc_paths as BP
sys.path.insert(0, BP.REPO)
from autogt import lane_gt
from scipy.spatial import cKDTree
HOME = os.path.expanduser("~"); D = f"{HOME}/exp_gcar_b123"
sp = json.load(open(f"{BP.ROOT}/data/splits.json"))
rows = np.asarray(sp["val"], int)
vt = json.load(open(f"{D}/feats/val_tokens.json"))
t0 = time.time()
gt = pickle.load(open(f"{D}/graph_gt_val.pkl", "rb"))
print("graph_gt_val", type(gt), len(gt), time.time() - t0)
g0 = gt[rows[0]]
print(type(g0), list(g0.keys()), len(g0["pts"]), g0["pts"][0].shape, g0["edges"][:5], g0.get("nodes") is None)
# agreement with the lane GT for some frames
import graph_eval as GE
res = []
for r in rows[::25]:
    g = lane_gt.load(vt[r])
    P, _, _ = GE.dense_points(gt[r])
    Q = np.concatenate([np.asarray(p["pts"]) for p in g["pieces"]])
    Q = Q[(np.abs(Q) < 48).all(1)]
    d1 = cKDTree(Q).query(P)[0]; d2 = cKDTree(P).query(Q)[0]
    res.append((float((d1 < 0.3).mean()), float((d2 < 0.3).mean())))
res = np.array(res)
print("old GT points within 0.3 of lane GT, lane GT points within 0.3 of old GT:", res.mean(0), res.min(0))
g = lane_gt.load(vt[rows[0]])
print(list(g.keys()))
print(len(g["pieces"]), [ (p["id"], p["piece"], p["kind"], len(p["pts"])) for p in g["pieces"][:6]])
print({k: (type(v), (len(v) if hasattr(v,'__len__') else v)) for k, v in g.items() if k != "pieces"})

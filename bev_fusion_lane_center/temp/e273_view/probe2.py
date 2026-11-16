import os, json, sys, numpy as np
HOME = os.path.expanduser("~"); R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, ".."))
from autogt import lane_gt
toks = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
sp = json.load(open(f"{R}/data/splits.json"))
pz = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")
held = lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature()))
print("signature", lane_gt.signature(), "held", len(held))
rows = np.array(sp["val"])
print("val rows", len(rows), "in store", sum(toks[r] in held for r in rows))
sc = pz["scene"][rows]
tab = os.path.join(HOME, "data/nuscenes_full/v1.0-trainval")
S = json.load(open(f"{tab}/scene.json")); SA = json.load(open(f"{tab}/sample.json"))
sname = {s["token"]: s["name"] for s in S}
tok2scene = {s["token"]: sname[s["scene_token"]] for s in SA}
names = {}
for r in rows:
    names.setdefault(int(pz["scene"][r]), tok2scene[toks[r]])
for k in sorted(names): print(k, names[k], int((sc == k).sum()), sum(toks[r] in held for r in rows[sc == k]))
print({k: (type(v), (len(v) if hasattr(v, '__len__') else v)) for k, v in sp.items()})

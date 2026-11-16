import json, os, sys, numpy as np
import bflc_paths as BP
sys.path.insert(0, BP.REPO)
from autogt import lane_gt
HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
sp = json.load(open(f"{BP.ROOT}/data/splits.json"))
print({k: len(v) for k, v in sp.items()})
vt = json.load(open(f"{D}/feats/val_tokens.json"))
print("val tokens", len(vt), vt[:2])
rows = np.asarray(sp["val"], int)
print("rows", rows[:10], rows.max())
sig = lane_gt.signature()
print("sig", sig)
d = os.path.join(lane_gt.DATA, sig)
held = lane_gt.held(d)
print("held", len(held))
toks = [vt[r] for r in rows]
print("unique tokens", len(set(toks)))
have = [t in held for t in toks]
print("VAL stored", sum(have), "of", len(toks))

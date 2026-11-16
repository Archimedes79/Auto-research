import os, json, numpy as np, sys
HOME = os.path.expanduser("~")
pz = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")
print({k: (pz[k].shape, pz[k].dtype) for k in pz.files})
toks = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
print(len(toks), toks[:2])
sp = json.load(open(os.path.join(os.environ["BFLC_ROOT"], "data/splits.json")))
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in sp.items()})
sc = pz["scene"]
print(np.unique(sc)[:20], len(np.unique(sc)))

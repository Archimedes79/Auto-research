import os
import pickle
import numpy as np

ROOT = os.environ["BFLC_ROOT"]
d = pickle.load(open(f"{ROOT}/runs/e312_p_s0/ends_val.pkl", "rb"))
print(len(d), type(d[0]))
f = d[10]
for k, v in f.items():
    v = np.asarray(v) if not isinstance(v, (str, int, float)) else v
    print(k, getattr(v, "shape", v), getattr(v, "dtype", ""))
print("valid", np.asarray(f["valid"]).astype(int))
print("kp_p", np.round(np.asarray(f["kp_p"]), 2))
g = pickle.load(open(f"{ROOT}/temp/jwiring/gt_ends_val.pkl", "rb"))
r = next(iter(g))
print({k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in g[r].items()})
print("ent_m sum, ext_m sum", g[r]["ent_m"].sum(), g[r]["ext_m"].sum())
n_valid = [int(np.asarray(x["valid"]).sum()) for x in d]
print("mean valid keypoints", np.mean(n_valid), "mean GT ends", np.mean([g[k]["ent_m"].sum() + g[k]["ext_m"].sum() for k in g]))

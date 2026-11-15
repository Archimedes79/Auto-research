import numpy as np, json, os
d = os.path.expanduser("~/exp_gcar_b123")
for s in ("train", "val"):
    p = np.load(f"{d}/poses_{s}.npz")
    print(s, {k: p[k].shape for k in p.files})
    prev = p["prev"]; sc = p["scene"]
    ok = prev >= 0
    print("  rows", len(prev), "with prev", int(ok.sum()), "prev in same scene", bool((sc[prev[ok]] == sc[ok]).all()), "prev<row", bool((prev[ok] < np.arange(len(prev))[ok]).all()))
sp = json.load(open("data/splits.json"))
print({k: len(v) for k, v in sp.items() if hasattr(v, "__len__")})
tr = set(sp["train"]) if "train" in sp else set()
p = np.load(f"{d}/poses_train.npz")
rows = np.asarray(sp["train"], int)
pv = p["prev"][rows]
print("train rows", len(rows), "prev in the train split:", int(np.isin(pv[pv >= 0], rows).sum()), "of", int((pv >= 0).sum()))

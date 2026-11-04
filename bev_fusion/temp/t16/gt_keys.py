import glob, numpy as np
f = sorted(glob.glob("bev_fusion/data/cache/gt/*_c643a7539d.npz"))[0]
g = np.load(f, allow_pickle=True)["gt"].item()
print(f.split("/")[-1], list(g))
c = g["cur"]
for k, v in c.items():
    if isinstance(v, dict):
        print(" ", k, {n: (a.shape, str(a.dtype)) if hasattr(a, "shape") else type(a).__name__ for n, a in list(v.items())[:14]})
    else:
        print(" ", k, getattr(v, "shape", type(v)))

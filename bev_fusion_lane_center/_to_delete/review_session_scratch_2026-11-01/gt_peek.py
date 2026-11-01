import os, sys, json, math, numpy as np
d = "data/cache/gt"
fs = sorted(f for f in os.listdir(d) if f.endswith("15efbde2d1.npz"))[:3]
z = np.load(os.path.join(d, fs[0]), allow_pickle=True)
print(fs[0], list(z.keys())[:40])
for k in z.keys():
    a = z[k]
    print(k, a.dtype, a.shape if a.dtype != object else type(a.item() if a.shape == () else a))

import glob, numpy as np, json
fs = sorted(glob.glob("data/cache/gt/*_6a7aa7864a*.npz"))[:1] or sorted(glob.glob("data/cache/gt/*.npz"))[:1]
print(fs)
d = np.load(fs[0], allow_pickle=True)["gt"].item()
print(d.keys())
for k in ("prev", "cur", "next"):
    if k in d:
        m = d[k]["meta"]
        print(k, m["sample"], m["scene"], m.get("dt"), [kk for kk in d[k] if kk not in ("meta",)][:6])

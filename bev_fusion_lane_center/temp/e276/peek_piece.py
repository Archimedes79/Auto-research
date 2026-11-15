import gzip, pickle, sys, os, glob
store = "/mnt/c/Auto-research/data/autogt/lane/0702a065c6"
f = sorted(glob.glob(store + "/*.pkl.gz"))[100]
g = pickle.load(gzip.open(f, "rb"))
print(type(g), list(g.keys())[:40])
p = g["pieces"][0]
print("piece keys", list(p.keys()))
for k, v in p.items():
    print(" ", k, type(v).__name__, (getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else v)))

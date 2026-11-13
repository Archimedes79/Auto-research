import gzip, pickle, glob, os
import numpy as np
d = os.path.expanduser("/mnt/c/Auto-research/data/autogt/lane/3576fa1f7e")
fs = sorted(glob.glob(d + "/*.pkl.gz"))
print(len(fs), "files")
g = pickle.load(gzip.open(fs[len(fs) // 2], "rb"))
print(type(g), list(g.keys()))
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in g.items()})
p = g["pieces"][0]
print("piece keys", list(p.keys()))
print({k: (type(v).__name__, getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else v)) for k, v in p.items()})
print("segments", p["segments"][:3])
print("event keys", list(g["events"][0].keys()), [e["type"] for e in g["events"]][:20])
print("lines", {k: (len(v), list(v[0].keys()) if v else None) for k, v in g["lines"].items()})
print("areas", {k: (len(v), list(v[0].keys()) if v else None) for k, v in g["areas"].items()})
print("roads", g["roads"][0] if g["roads"] else None)
print("params", g["params"], g.get("frame"), g.get("half"), g.get("location"))
print({k: g["meta"] if False else None for k in []})

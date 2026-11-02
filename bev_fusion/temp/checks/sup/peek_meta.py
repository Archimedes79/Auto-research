import glob, numpy as np
f = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[0]
g = np.load(f, allow_pickle=True)["gt"].item()
print(f, list(g.keys()))
m = g["cur"]["meta"]
print({k: (type(v).__name__, (v if not isinstance(v, (list, np.ndarray)) else len(v))) for k, v in m.items()})
print(m["boxes"][0].keys() if m["boxes"] else None)
print("prev dt", g["prev"]["meta"]["dt"], "steps0", g["prev"]["meta"]["steps"][:2], "next dt", g["next"]["meta"]["dt"])
print({k: getattr(v, "shape", None) for k, v in g["cur"].items() if k != "meta"})

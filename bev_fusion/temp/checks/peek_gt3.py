import glob, numpy as np
f = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[0]
g = np.load(f, allow_pickle=True)["gt"].item()
print("top keys", list(g.keys()))
for part in ("prev", "next"):
    m = g[part]["meta"]
    print(part, "meta keys", list(m.keys()))
    if "steps" in m: print("  steps", m["steps"][:3], "... n", len(m["steps"]))
print("box keys", list(g["cur"]["meta"]["boxes"][0].keys()))
print("velocity range", float(np.abs(g["cur"]["velocity"]).max()), "vehicle cells", int((g["cur"]["vehicle"] > 0.5).sum()))

import os, math, numpy as np
d = "data/cache/gt"
fs = sorted(f for f in os.listdir(d) if f.endswith("15efbde2d1.npz"))
g = np.load(os.path.join(d, fs[0]), allow_pickle=True)["gt"].item()
print("top keys", list(g.keys()))
s = g.get("cur", g)
print("cur keys", list(s.keys()) if isinstance(s, dict) else type(s))
m = s["meta"]; print("meta keys", list(m.keys()))
print("box0", {k: (v if not hasattr(v, 'shape') or v.size < 10 else v.shape) for k, v in m["boxes"][0].items()})
# angle agreement over many files
rng = np.random.default_rng(0)
agree, n_mov, speeds, inst = [], 0, [], []
for f in rng.choice(fs, 300, replace=False):
    g = np.load(os.path.join(d, f), allow_pickle=True)["gt"].item()
    for side in ("cur",):
        s = g[side] if side in g else g
        for b in s["meta"]["boxes"]:
            if b["group"] != "vehicle":
                continue
            v = np.asarray(b["vel"], float)
            sp = float(np.hypot(*v)) if np.all(np.isfinite(v)) else float("nan")
            speeds.append(sp)
            if sp > 2.0:
                n_mov += 1
                ang = math.atan2(v[1], v[0]) - float(b["yaw"])
                agree.append(abs(math.cos(ang)))
sp = np.array(speeds)
print("vehicles", len(sp), "nan", int(np.isnan(sp).sum()), "moving>1", float(np.nanmean(sp > 1)), "mean speed moving", float(np.nanmean(sp[sp > 1])))
a = np.array(agree)
print("moving>2", n_mov, "|cos(vel angle - yaw)| median", float(np.median(a)), "share > 0.9", float((a > 0.9).mean()))

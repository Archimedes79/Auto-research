import os, numpy as np, collections
d = "data/cache/gt"
fs = sorted(f for f in os.listdir(d) if f.endswith("15efbde2d1.npz"))
rng = np.random.default_rng(1)
c = collections.Counter(); flags = collections.Counter(); src = collections.Counter()
cells_obj = cells_mov = 0; vabs = []
for f in rng.choice(fs, 400, replace=False):
    s = np.load(os.path.join(d, f), allow_pickle=True)["gt"].item()["cur"]
    for b in s["meta"]["boxes"]:
        c[(b["group"], bool(b.get("vel_known", True)), bool(b.get("observed", True)))] += 1
        flags[(b["group"], b.get("vel_flag"))] += 1
        src[b.get("vel_source")] += 1
    inst = s["instance"] >= 0
    w = np.maximum(s["camera"] * (1 - s["occluded"]), s["radar"]) * (1 - np.clip(s["ignore"], 0, 1))
    sel = inst & (w > 0)
    v = s["velocity"][:, sel]
    sp = np.hypot(*v)
    vv = np.where(sp < 1.0, 0, 1) * v
    cells_obj += sel.sum(); cells_mov += int((sp >= 1).sum())
    vabs.append(np.abs(vv).sum())
print("(group, vel_known, observed) counts", sorted(c.items()))
print("flags", sorted(flags.items(), key=lambda kv: -kv[1])[:15])
print("sources", src.most_common())
print("motion obj cells", cells_obj, "moving share", cells_mov / max(1, cells_obj), "zero-pred L1 per comp", sum(vabs) / (2 * cells_obj))

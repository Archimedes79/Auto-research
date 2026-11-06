"""inspect scene_records pickles (CPU, read only)"""
import pickle, sys, json
p = sys.argv[1]
r = pickle.load(open(p, "rb"))
print("keys", list(r))
m = r.get("meta")
if m:
    print("meta keys", list(m), {k: (v if not isinstance(v, (list, dict)) else (type(v).__name__, len(v))) for k, v in m.items()})
    if m.get("frames"):
        print("frames[:3]", m["frames"][:3])
d = r["det"]
scs = sorted(d)
print("n scenes", len(scs), scs[:3])
s = scs[0]
print("dts", list(d[s]))
dt0 = list(d[s])[0]
fr = d[s][dt0]
print("frames in scene", len(fr), type(fr[0]), len(fr[0]))
dets, gts = fr[0]
print("det[0]", dets[0] if dets else None)
print("gt[0]", gts[0] if gts else None)
print("n gt frame0", len(gts), "n det", len(dets))
print("total frames", sum(len(d[s][dt0]) for s in scs))

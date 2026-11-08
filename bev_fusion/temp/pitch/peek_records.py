import pickle, sys
d = pickle.load(open("runs/t22_B_s0/eval/scene_records_n1000.pkl" if len(sys.argv) < 2 else sys.argv[1], "rb"))
print(type(d), list(d)[:6] if isinstance(d, dict) else len(d))
m = d.get("meta"); print("meta keys", list(m)[:6], "frames", len(m.get("frames", [])) if m else None, m.get("gt_signature") if m else None)
for k in d:
    if k != "meta":
        v = d[k]; print(k, type(v), list(v)[:3] if hasattr(v, "keys") else len(v))
        break

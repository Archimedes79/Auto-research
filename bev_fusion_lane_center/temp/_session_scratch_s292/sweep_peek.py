import json, os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/e345/e312_p_s1/sweep.json")
d = json.load(open(p))
print("top keys", list(d)[:12])
for k, v in d.items():
    if isinstance(v, (list, dict)):
        print(k, type(v).__name__, len(v))
    else:
        print(k, v)
ps = d.get("passes") or d.get("points") or {}
items = list(ps.values()) if isinstance(ps, dict) else ps
if items:
    last = items[-1]
    print("last pass keys", list(last)[:15] if isinstance(last, dict) else last)
    for k in ("sec", "seconds", "time", "gpu_mb", "peak_mb"):
        vals = [x.get(k) for x in items if isinstance(x, dict) and x.get(k) is not None]
        if vals:
            print(k, "n", len(vals), "mean", round(sum(vals) / len(vals), 1), "max", max(vals))

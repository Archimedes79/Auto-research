p = "evaluation/bev_suite.py"
t = open(p, encoding="utf-8").read()
a = '''        pv.setdefault("data", {})["use"] = data_use(a.split, store, avail)'''
b = '''        pv.setdefault("data", {})["use"] = data_use(a.split, store, avail)
        ver = (_read_json(cfg) or {}).get("version")                          # the collector leaves the raw-data version "unknown" without a GT store that names it
        if ver and pv["data"].get("dataset") in (None, "unknown"):
            pv["data"]["dataset"] = ver'''
assert t.count(a) == 1
t = t.replace(a, b)
a = '''"(71 GFLOPs per frame as stated in training/feature_cache.py; the BEVFusion heads of a dual chain are not counted)=71"]'''
b = '''"(71 GFLOPs per frame as stated in training/feature_cache.py, unit not verified: if they are MACs the figure doubles; the BEVFusion heads of a dual chain are not counted)=71"]'''
assert t.count(a) == 1, t.count(a)
t = t.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched")

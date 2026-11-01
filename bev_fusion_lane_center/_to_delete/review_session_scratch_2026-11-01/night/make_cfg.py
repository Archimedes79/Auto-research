"""derive a bev_fusion config from another: make_cfg.py <src.json> <dst.json> key=value ... (value as JSON; model.x / loss.x for nested keys)"""
import json
import sys

src, dst, *kv = sys.argv[1:]
c = json.load(open(src))
for item in kv:
    k, v = item.split("=", 1)
    v = json.loads(v)
    d = c
    *path, last = k.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = v
json.dump(c, open(dst, "w"), indent=1)
print("wrote", dst, {i.split("=")[0]: c.get(i.split("=")[0], "nested") for i in kv})

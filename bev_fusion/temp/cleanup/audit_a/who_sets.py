"""audit scratch: which stored configs set field=value. usage: who_sets.py model|loss field [value-json]"""
import glob, json, sys
ROOT = "/mnt/c/Auto-research/bev_fusion"
sub, field = sys.argv[1], sys.argv[2]
want = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
n = 0
for p in sorted(glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/training/configs/presets/*.json") + glob.glob(f"{ROOT}/runs/*/config.json")):
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    d = (c.get(sub) or {}) if isinstance(c, dict) else {}
    if field in d and (want is None or d[field] == want):
        n += 1
        print(p.replace(ROOT + "/", ""), json.dumps(d[field]))
print("total", n)

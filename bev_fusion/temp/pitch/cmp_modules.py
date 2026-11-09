import json
import math

new = "temp/pitch/modules_check/"
old = "runs/t16_small_s0/eval/modules/"
worst = 0.0
for f in ("modules.json", "response.json", "static_check.json"):
    a, b = json.load(open(old + f)), json.load(open(new + f))
    if f == "static_check.json":
        keys = [(r["frame"], r["level"]) for r in a]
        bk = {(r["frame"], r["level"]): r for r in b}
        miss = [k for k in keys if k not in bk]
        n = 0
        for r in a:
            o = bk.get((r["frame"], r["level"]))
            if not o:
                continue
            for k, v in r.items():
                if isinstance(v, (int, float)) and o.get(k) is not None:
                    d = abs(v - o[k]) / (abs(v) + 1e-9)
                    worst = max(worst, d if abs(v) > 1e-6 else abs(v - o[k]))
                    n += 1
        print(f, "rows", len(a), "vs", len(b), "missing in new:", miss, "numbers compared", n)
        continue
    only_old = [k for k in a if k not in b]
    only_new = [k for k in b if k not in a]
    n = 0
    for k in a:
        if k not in b:
            continue
        va, vb = a[k], b[k]
        if isinstance(va, dict):
            for kk, x in va.items():
                y = vb.get(kk)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                    d = abs(x - y) / (abs(x) + 1e-9) if abs(x) > 1e-6 else abs(x - y)
                    if d > 1e-3:
                        print("  DIFF", f, k, kk, x, y)
                    worst = max(worst, d)
                    n += 1
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(va - vb) / (abs(va) + 1e-9) if abs(va) > 1e-6 else abs(va - vb)
            if d > 1e-3:
                print("  DIFF", f, k, va, vb)
            worst = max(worst, d)
            n += 1
    print(f, "keys old", len(a), "new", len(b), "only in old:", only_old[:6], "only in new:", only_new[:8], "numbers compared", n)
print("worst relative difference over everything compared:", worst)

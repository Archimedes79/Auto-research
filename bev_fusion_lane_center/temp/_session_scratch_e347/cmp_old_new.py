import json, sys
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); o = json.load(open("score_old_new/e332_ma_s0/old.json")); n = json.load(open("score_old_new/e332_ma_s0/" + (sys.argv[1] if len(sys.argv) > 1 else "new") + ".json"))
bad = []; cnt = [0]
def cmp(a, b, path):
    if isinstance(a, dict):
        for k in a:
            if k not in b: bad.append((path + "/" + k, "missing in new")); continue
            cmp(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        if len(a) != len(b): bad.append((path, "len")); return
        for i, (x, y) in enumerate(zip(a, b)): cmp(x, y, f"{path}[{i}]")
    else:
        cnt[0] += 1
        if a != b: bad.append((path, a, b))
cmp(o, n, "")
print("old keys compared:", cnt[0], "differences:", len(bad)); print(bad[:10])
print("new top-level keys not in old:", sorted(set(n) - set(o)))
print("frames", n["frames"], "readout", n["readout"])
for who in ("ours", "ours_alt"):
    print(who, json.dumps(n["gate"][who]["at_best"], indent=None)[:400])
r = n["thr_sweep"]["ours"]["ent"]["rows"][1]; print("sweep row thr 0.3 ours ent:", r)

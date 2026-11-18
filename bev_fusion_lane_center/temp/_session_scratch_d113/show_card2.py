import json, os, sys
R = os.environ["BFLC_ROOT"]
c = json.load(open(os.path.join(R, "runs", sys.argv[1], sys.argv[2])))
print(list(c["info"].keys()))
for k, v in c["info"].items():
    if isinstance(v, dict) and ("topo" in k or "junction" in k or "lanes" in k):
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if not isinstance(y, (list, dict))})
pj = c["info"].get("paired_junction_vs_ref_run") or {}
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pj.items()} if pj else "")
for k in ("paired_vs_ref_run",):
    for m in ("topo_F1", "lanes_found", "junction_F1"):
        if m in c["info"][k]:
            print(m, {x: round(y, 4) for x, y in c["info"][k][m].items() if x in ("diff", "lo", "hi")})

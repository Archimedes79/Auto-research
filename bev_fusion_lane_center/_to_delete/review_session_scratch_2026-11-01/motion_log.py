import json, sys
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    tr = [r for r in rows if "val" not in r]
    print("==", run, len(tr))
    for r in tr[::10]:
        print(r["step"], round(r.get("group.motion", float("nan")), 4), "s", round(r.get("s.motion", float("nan")), 3),
              "s.obj", round(r.get("s.objects", float("nan")), 3), "gn", round(r.get("grad_norm", float("nan")), 1))

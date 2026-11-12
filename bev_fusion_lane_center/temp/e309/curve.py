import json, sys, numpy as np
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"../../runs/{run}/log.jsonl") if l.strip().startswith("{")]
    rows = [r for r in rows if "loss" in r or "step" in r]
    print(run, len(rows), "keys", list(rows[-1].keys())[:12])
    key = "total"
    st = np.array([r.get("step", i) for i, r in enumerate(rows)]); ls = np.array([r[key] for r in rows], float)
    for a, b in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3001)):
        m = (st >= a) & (st < b)
        if m.any(): print(f"  steps {a}-{b}: mean {key} {ls[m].mean():.4f}")

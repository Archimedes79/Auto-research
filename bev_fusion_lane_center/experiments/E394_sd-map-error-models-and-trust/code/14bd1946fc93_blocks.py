import json, os
import numpy as np
root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for arm, key in (("p0", "curve"), ("p0n", "node"), ("p2", "curve")):
    out = []
    for s in (0, 1):
        rows = [json.loads(l) for l in open(os.path.join(root, "runs", "e395_%s_s%d" % (arm, s), "log.jsonl"))]
        v = {r["step"]: r[key] for r in rows if "step" in r and "total" in r}
        a = np.mean([v[t] for t in range(100, 1001, 100)]); b = np.mean([v[t] for t in range(1100, 2001, 100)])
        out.append("%.4f" % (b / a))
    print(arm, out)

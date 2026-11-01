"""E240: per-seed values and seed SDs of the three finished pairs (VAL 873 cards)."""
import json, os
R = os.environ.get("BFLC_ROOT", ".")
import numpy as np
ks = ("Q", "lanes_found", "R@1_far", "topo_F1", "F1@1.0", "vert100")
v = {}
for arm in ("rc", "eq"):
    v[arm] = {k: [] for k in ks}
    for s in range(4):
        d = json.load(open(f"{R}/runs/eval/e24_j1{arm}_s{s}/val.json", encoding="utf-8-sig"))
        m = d.get("metrics", d)
        for k in ks:
            v[arm][k].append(m[k])
for k in ks:
    a, c = np.array(v["rc"][k]), np.array(v["eq"][k])
    dd = a - c
    print(f"{k:12} rc {np.round(a,4).tolist()} mean {a.mean():.4f} sd {a.std(ddof=1):.4f} | eq {np.round(c,4).tolist()} mean {c.mean():.4f} sd {c.std(ddof=1):.4f} | diff mean {dd.mean():+.4f} sd {dd.std(ddof=1):.4f} min {dd.min():+.4f} max {dd.max():+.4f}")

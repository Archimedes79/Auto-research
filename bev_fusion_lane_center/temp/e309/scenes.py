import os, pickle, sys
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", "perframe_s0.pkl"), "rb"))
by = {}
for r in PF:
    by.setdefault(r["scene"], []).append(r)
rows = []
for s, v in by.items():
    b = np.mean([x["B"]["num"]["topo_F1"] for x in v]); e = np.mean([x["e312"]["num"]["topo_F1"] for x in v])
    rows.append((s, len(v), b, e, e - b, min(x["row"] for x in v)))
for s, n, b, e, d, r0 in sorted(rows, key=lambda t: -t[4]):
    print(f"{s} frames {n:3d} B {b:.3f} E312 {e:.3f} delta {d:+.3f} first row {r0}")

import json
json.dump([dict(scene=str(a), n=int(b), B=float(c), e312=float(d), first_row=int(f)) for a, b, c, d, e, f in sorted(rows, key=lambda t: int(t[0]))], open(os.path.join(ROOT, "temp", "e309", "scenes_s0.json"), "w"), indent=1)

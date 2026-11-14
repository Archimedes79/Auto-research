import sys
import numpy as np

sys.path.insert(0, "../../tools/viz")
import qgraph_view as QV

tok = QV._frame_token(sys.argv[1] if len(sys.argv) > 1 else "scene-0096:10")
g = QV.load_gt(tok)
print("token", tok, "keys", list(g.keys()))
P = g["pieces"]
print("pieces", len(P), "kinds", {k: sum(1 for p in P if p["kind"] == k) for k in ("lane", "connector", "mixed")})
for p in P[:6]:
    print({k: (v if not hasattr(v, "shape") else v.shape) for k, v in p.items() if k not in ("tokens",)})
print("events", len(g["events"]))
from collections import Counter
print(Counter(e["type"] for e in g["events"]))
for e in g["events"][:4]:
    print({k: (np.round(v, 2).tolist() if hasattr(v, "shape") else v) for k, v in e.items()})
print("roads", [(r["id"], r["kind"], r["lane_max"], len(r["pieces"]), r["succ"]) for r in g["roads"]][:12])
print("areas inter", [(a["id"], round(a["area"], 0)) for a in g["areas"]["intersection"]])
print("tier keys", [k for k in P[0].keys()])
print("ego_path", None if "ego_path" not in g else g["ego_path"]["pts"].shape)

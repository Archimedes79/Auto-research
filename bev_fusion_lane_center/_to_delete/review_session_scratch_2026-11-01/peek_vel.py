import json
import numpy as np
o = json.load(open("/mnt/c/Auto-research/bev_fusion/runs/t10_dual_s0/prototype/objects.json"))
for key in ("objects", "forecast_0.5s"):
    v = np.array([b["velocity"] for f in o["scenes"]["scene-0003"] for b in f[key]])
    print(key, len(v), "distinct", len({tuple(x) for x in v.tolist()}), "max |v|", np.abs(v).max(0), "mean |v|", np.abs(v).mean(0))

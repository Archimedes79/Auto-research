import json
import numpy as np
d = "/mnt/c/Auto-research/bev_fusion/runs/t10_dual_s0/prototype/"
o = json.load(open(d + "objects.json"))
for f in o["scenes"]["scene-0003"]:
    b = f["objects"]
    hi = [x for x in b if x["score"] >= 0.3]
    print(f["index"], f["box_source"], len(b), "boxes;", len(hi), ">= 0.3:", sorted({x["class"] for x in hi}),
          "forecast", len(f["forecast_0.5s"]), "e.g.", hi[0] if hi else None)
m = np.load(d + "maps_scene-0003.npz")
print(list(m["layers"]), m[str(m["layers"][0])].shape, json.loads(str(m["grid"])))
print({k: round(float(m[k].mean()), 3) for k in m["layers"]})

import numpy as np, json
z = np.load("runs/t10_dual_s0/prototype/maps_scene-0003.npz", allow_pickle=True)
print({k: (z[k].shape, z[k].dtype) for k in z.files})
o = json.load(open("runs/t10_dual_s0/prototype/objects.json"))
fr = o["scenes"]["scene-0003"]
print(len(fr), [ (f["index"], f["box_source"], len(f["objects"]), None if f["forecast_0.5s"] is None else len(f["forecast_0.5s"])) for f in fr])
print(json.load(open("runs/t10_dual_s0/prototype/summary.json")).get("subject", {}).get("device"))

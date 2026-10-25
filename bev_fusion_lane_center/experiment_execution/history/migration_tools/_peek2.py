import json
import os

import numpy as np

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
s = json.load(open(f"{R}/data/splits.json"))
print(type(s).__name__, list(s.keys())[:12] if isinstance(s, dict) else len(s))
for k, v in list(s.items())[:12]:
    print("  ", k, type(v).__name__, (len(v) if hasattr(v, "__len__") else v), (v[:5] if isinstance(v, list) else (list(v.items())[:2] if isinstance(v, dict) else "")))
H = os.path.expanduser("~/exp_gcar_b123")
for f in ("poses_val.npz", "gtK_val.npz"):
    z = np.load(f"{H}/{f}", allow_pickle=True)
    print(f, {k: (z[k].shape, z[k].dtype) for k in z.files})

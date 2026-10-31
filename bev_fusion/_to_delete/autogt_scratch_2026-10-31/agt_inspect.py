"""scratch: what a stored autoGT keyframe really contains (keys, shapes, dtypes, ranges, meta, box fields)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from autogt.cli import load_keyframe  # noqa: E402

for sig in ("15efbde2d1", "7fe1149566", "fefd519974"):
    d = os.path.join("data/autogt", sig)
    n = len(glob.glob(os.path.join(d, "*.npz")))
    idx = json.load(open(os.path.join(d, "index.json"))) if os.path.exists(os.path.join(d, "index.json")) else None
    print(sig, "npz files:", n, "| scene pickles:", len(glob.glob(os.path.join(d, "_scene_*.pkl"))),
          "| index:", None if idx is None else {k: idx[k] for k in ("version", "res", "keyframes")})
d = "data/autogt/15efbde2d1"
f = sorted(glob.glob(os.path.join(d, "*.npz")))[100]
tok = os.path.basename(f)[:-4]
g = load_keyframe(d, tok)
print("token", tok)
for k, v in g.items():
    if k == "meta":
        continue
    v = np.asarray(v)
    print(f"{k:18s} {str(v.shape):14s} {v.dtype}  min {v.min():9.3f} max {v.max():9.3f}")
m = g["meta"]
print("meta keys:", list(m.keys()))
for k, v in m.items():
    if isinstance(v, (int, float, str)):
        print("  ", k, "=", v)
    elif isinstance(v, np.ndarray):
        print("  ", k, v.shape, v.dtype)
    elif isinstance(v, dict):
        print("  ", k, "dict", {kk: (len(vv) if hasattr(vv, "__len__") else vv) for kk, vv in v.items()})
    elif isinstance(v, list):
        print("  ", k, "list", len(v))
b = m["boxes"][0]
print("box keys:", {k: (type(v).__name__ if not isinstance(v, np.ndarray) else (v.shape, v.dtype)) for k, v in b.items()})
print("box example:", {k: (v if not isinstance(v, np.ndarray) else np.round(v, 2).tolist()) for k, v in b.items() if k in ("group", "category", "observed", "stationary", "vel", "vel_source", "vel_flag", "vel_sigma", "vel_known", "shift", "shift_s", "center", "size", "yaw", "instance")})

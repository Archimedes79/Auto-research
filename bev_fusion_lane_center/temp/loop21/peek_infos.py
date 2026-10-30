"""Loop 21 one-off: can the VAL rows be matched to camera images? poses_val.npz keys, the infos pickle, the split rows."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV  # noqa: E402

z = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz", allow_pickle=True)
print({k: (z[k].shape, str(z[k].dtype)) for k in z.files})
sp = json.load(open(f"{EV.R}/data/splits.json"))
print("split keys", list(sp), "val rows", len(sp["val"]), sp["val"][:5], "max", max(sp["val"]))
for fn in ("nuscenes_infos_val.pkl", "nuscenes_infos_val_crn.pkl"):
    d = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/{fn}", "rb"))
    infos = d["infos"] if isinstance(d, dict) else d
    print(fn, type(d).__name__, len(infos), "keys", list(infos[0])[:14])
    i0 = infos[0]
    for k in ("scene_token", "token", "timestamp", "sample_token"):
        if k in i0:
            print("  ", k, i0[k])
    ci = i0.get("cam_infos") or i0.get("cams")
    if ci:
        c = ci["CAM_FRONT"]
        print("  CAM_FRONT keys", list(c)[:8], str(c.get("filename") or c.get("data_path") or c.get("image_path"))[:120])

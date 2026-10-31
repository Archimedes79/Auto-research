import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E
import evaluate as EV
E.load("recall")
S = E.S
r = S["raw"][105]
for k in ("pos","th_travel","flag","pi","nu","valid","parent","level"):
    v = np.asarray(r[k]); print(k, v.shape, v.dtype, float(v.min()), float(v.max()), "mean", float(v.mean()), "n>0.7", int((v>0.7).sum()))
print("level counts", np.unique(r["level"], return_counts=True))
print("parent[:20]", r["parent"][:20])
print("tau", S["tau"], "asm", S["asm"])
print("link_p", type(r["link_p"]), len(r["link_p"]))
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
ci = infos[0]["cam_infos"]["CAM_FRONT"]
print({k: (np.asarray(v).shape if hasattr(v,'__len__') and not isinstance(v,str) else v) for k, v in ci.items()})
print({k: v for k, v in ci.items() if k in ("sensor2ego_translation","sensor2ego_rotation","intrinsics","intrinsic","sensor2lidar_rotation","sensor2lidar_translation")})
print(infos[0].keys())

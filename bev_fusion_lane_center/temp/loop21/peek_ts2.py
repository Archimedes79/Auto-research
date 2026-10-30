import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
T = np.array([i["timestamp"] for i in infos], dtype=np.float64) / 1e6
ts = pz["ts"]
d = np.array([np.min(np.abs(T - x)) for x in ts])
print("poses ts range", ts.min(), ts.max(), "infos range", T.min(), T.max())
print("closest-info diff: min %.4f median %.4f max %.4f" % (d.min(), np.median(d), d.max()), "n<1e-3:", int((d < 1e-3).sum()), "n<0.05:", int((d < 0.05).sum()))
print("sorted ts equal?", np.allclose(np.sort(ts), np.sort(T)))
print("poses ts[:6]", ts[:6], "diff", np.diff(ts[:6]))

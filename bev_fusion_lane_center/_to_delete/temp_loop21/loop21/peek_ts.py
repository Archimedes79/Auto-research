import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
inf2 = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val.pkl", "rb"))["infos"]
print(pz["ts"][:4], [i["timestamp"] for i in infos[:4]], [i["timestamp"] for i in inf2[:4]])
t_old = np.array([i["timestamp"] for i in inf2], dtype=np.float64)
t_crn = np.array([i["timestamp"] for i in infos], dtype=np.float64)
print("poses ts == old infos/1e6:", np.allclose(pz["ts"], t_old / 1e6), "| == crn:", np.allclose(pz["ts"], t_crn / 1e6), "| old==crn", np.array_equal(t_old, t_crn))

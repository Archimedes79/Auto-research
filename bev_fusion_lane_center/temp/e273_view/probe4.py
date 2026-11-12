import os, pickle, time, numpy as np
HOME = os.path.expanduser("~")
t0 = time.time()
infos = pickle.load(open(f"{HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
print(len(infos), time.time() - t0)
i = infos[0]
print({k: (type(v).__name__) for k, v in i.items()})
c = i["cam_infos"]["CAM_BACK"]
print(c.keys())
for k, v in c.items():
    print(k, v if not isinstance(v, dict) else {kk: (vv if np.size(vv) < 20 else '...') for kk, vv in v.items()})

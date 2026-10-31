import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
i = infos[0]
print(i["cam_infos"]["CAM_FRONT"]["calibrated_sensor"])
print(i["lidar_infos"].keys() if isinstance(i["lidar_infos"], dict) else type(i["lidar_infos"]))
l = i["lidar_infos"]["LIDAR_TOP"]
print({k: v for k, v in l.items() if k in ("calibrated_sensor","ego_pose")})

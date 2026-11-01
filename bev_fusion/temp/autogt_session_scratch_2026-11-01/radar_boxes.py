"""scratch (D36): which annotated objects carry radar points? nuScenes mini, annotations inside the grid (0 <= x < 60 m, |y| < 26 m in the ego frame)
of their keyframe: share with num_radar_pts == 0 by group and by moving (> 1 m/s) / not.   Run: wsl bash autogt/run_nice.sh <this file>"""
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.insert(0, ".")
from autogt.nuscenes_adapter import category_group  # noqa: E402

nusc = NuScenes(version="v1.0-mini", dataroot=os.path.expanduser("~/data/nuscenes"), verbose=False)
rows = []
for sample in nusc.sample:
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    R = Quaternion(pose["rotation"]).rotation_matrix
    for atok in sample["anns"]:
        a = nusc.get("sample_annotation", atok)
        p = R.T @ (np.array(a["translation"]) - np.array(pose["translation"]))
        if not (0 <= p[0] < 60 and abs(p[1]) < 26):
            continue
        v = nusc.box_velocity(atok)
        grp = category_group(a["category_name"]) if "category_name" in a else category_group(nusc.get("instance", a["instance_token"])["category_token"] and nusc.get("category", nusc.get("instance", a["instance_token"])["category_token"])["name"])
        speed = float(np.hypot(*v[:2])) if np.isfinite(v).all() else float("nan")
        rows.append((grp or "other", speed, a["num_radar_pts"], a["num_lidar_pts"], float(np.hypot(p[0], p[1]))))
print("annotations inside the grid:", len(rows))
print(f"{'group':14s}{'moving':>8s}{'n':>7s}{'no radar pt':>13s}{'no lidar+radar':>16s}{'median radar pts':>18s}")
for g in sorted({r[0] for r in rows}):
    for mv in (True, False):
        s = [r for r in rows if r[0] == g and np.isfinite(r[1]) and ((r[1] > 1.0) == mv)]
        if len(s) >= 5:
            print(f"{g:14s}{str(mv):>8s}{len(s):7d}{np.mean([r[2] == 0 for r in s]):13.0%}{np.mean([r[2] + r[3] == 0 for r in s]):16.0%}{np.median([r[2] for r in s]):18.0f}")
near = [r for r in rows if r[4] < 30]
far = [r for r in rows if r[4] >= 30]
print(f"no radar point: {np.mean([r[2] == 0 for r in near]):.0%} of the objects closer than 30 m, {np.mean([r[2] == 0 for r in far]):.0%} of those beyond 30 m")

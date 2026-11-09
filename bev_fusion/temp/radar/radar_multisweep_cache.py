"""One-off: multi-sweep radar detections of every nuScenes-mini keyframe, ego-aligned to the camera time, cached for tests.

For each keyframe: output time t_ref = timestamp of the CAM_FRONT image (its ego pose = the output vehicle frame). For each of the
5 radars every sweep with -0.1 s <= t_ref - t_sweep <= 0.5 s is taken (the sweep chain through sample_data 'prev'). Per detection:
  x, y   position in the output vehicle frame (rigid ego alignment by the ego poses only: no Doppler shift, no object motion)
  r      range from the sensor (horizontal), psi  the ORIGINAL ray direction, rotated rigidly into the output frame
         (psi = direction of the sensor-frame ray, through sensor mounting, ego pose at the sweep and ego pose at t_ref)
  vr     ego-compensated radial velocity (vx_comp, vy_comp projected on the ray), rcs, dt = t_ref - t_sweep (s), sid sensor id
  scene  scene index (for splits)
Output: temp/radar/cache/mini_multisweep.pt  (list of per-frame dicts of float32 tensors)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_multisweep_cache.py
"""
import os

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]


def T(q, t):
    M = np.eye(4)
    M[:3, :3] = Quaternion(q).rotation_matrix
    M[:3, 3] = t
    return M


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot=os.path.expanduser("~/data/nuscenes"), verbose=False)
    RadarPointCloud.disable_filters()
    scene_idx = {s["token"]: i for i, s in enumerate(nusc.scene)}
    frames = []
    for sample in nusc.sample:
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        t_ref = cam["timestamp"]
        pe = nusc.get("ego_pose", cam["ego_pose_token"])
        T_ref_inv = np.linalg.inv(T(pe["rotation"], pe["translation"]))             # global -> output ego frame
        cols = {k: [] for k in ["x", "y", "r", "psi", "vr", "rcs", "dt", "sid"]}
        for sid, ch in enumerate(RADARS):
            sd = nusc.get("sample_data", sample["data"][ch])
            while True:
                dt = (t_ref - sd["timestamp"]) / 1e6
                if dt > 0.5:
                    break
                if dt >= -0.1:
                    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                    ego = nusc.get("ego_pose", sd["ego_pose_token"])
                    M = T_ref_inv @ T(ego["rotation"], ego["translation"]) @ T(cal["rotation"], cal["translation"])  # sensor -> output ego
                    P = RadarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"])).points
                    if P.shape[1]:
                        pts = M @ np.vstack([P[0], P[1], np.zeros_like(P[0]), np.ones_like(P[0])])
                        phi = np.arctan2(P[1], P[0])
                        ray = M[:3, :3] @ np.vstack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
                        cols["x"].append(pts[0]); cols["y"].append(pts[1])
                        cols["r"].append(np.hypot(P[0], P[1])); cols["psi"].append(np.arctan2(ray[1], ray[0]))
                        cols["vr"].append(P[8] * np.cos(phi) + P[9] * np.sin(phi)); cols["rcs"].append(P[5])
                        cols["dt"].append(np.full(P.shape[1], dt)); cols["sid"].append(np.full(P.shape[1], sid))
                if not sd["prev"]:
                    break
                sd = nusc.get("sample_data", sd["prev"])
        fr = {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in cols.items()}
        fr["scene"] = scene_idx[sample["scene_token"]]
        frames.append(fr)
    os.makedirs("temp/radar/cache", exist_ok=True)
    torch.save(frames, "temp/radar/cache/mini_multisweep.pt")
    n = [len(f["x"]) for f in frames]
    print(f"{len(frames)} frames, detections per frame (all 5 radars, window -0.1..0.5 s): median {int(np.median(n))}, max {max(n)}")

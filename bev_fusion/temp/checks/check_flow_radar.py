"""Which object-velocity estimator agrees best with the radar Doppler? (2026-10-30, nuScenes mini, all 404 keyframes)

For every annotated moving object (any estimator says > 1 m/s) the radar detections inside its box (5 radars, ego frame,
box grown by 0.5 m) carry an ego-compensated radial velocity d (the (vx_comp, vy_comp) of the pcd, which the radar driver
makes collinear with the line of sight). An estimator's world velocity v, rotated into ego axes, predicts d_hat = r_hat . v.
Reported per estimator: RMSE and median |d_hat - d| over all such points, share of points within 0.5 m/s, and the same for
a zero-velocity null. The radar sees only the radial component: tangential errors stay invisible.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_flow_radar.py
"""
import collections
import os
import sys

import numpy as np
from matplotlib.path import Path
from nuscenes.utils.data_classes import RadarPointCloud

sys.path.insert(0, "training")
from gt_bev import NuScenesGT, pose_matrix  # noqa: E402
from gt_flow import instance_tracks, object_velocity  # noqa: E402
from lift_pd import Grid  # noqa: E402

METHODS = ["dev", "dev+h", "fit1", "fit1+h", "fit2", "fit2+h"]
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")
nusc = src.nusc
G = Grid()
res = {m: [] for m in METHODS}
res["null"] = []
meta_rows = []                                    # (speed, number of points, scene)
for sc in nusc.scene:
    tracks = instance_tracks(src, sc["token"])
    idx_of = {r["atok"]: (inst, j) for inst, tr in tracks.items() for j, r in enumerate(tr)}
    tok = sc["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        gt = src.keyframe_gt(tok, G)
        T_e2g = gt["meta"]["T_e2g"]
        R_g2e = np.linalg.inv(T_e2g)[:2, :2]
        pts = []                                  # (x, y, d, rx, ry) in the ego frame
        for ch, sdtok in s["data"].items():
            if not ch.startswith("RADAR"):
                continue
            sd = nusc.get("sample_data", sdtok)
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            T = pose_matrix(cs)
            pc = RadarPointCloud.from_file(os.path.join(root, sd["filename"])).points
            P = (T[:3, :3] @ pc[:3]).T + T[:3, 3]
            V = (T[:3, :3] @ np.vstack([pc[8], pc[9], np.zeros(pc.shape[1])])).T
            r = P[:, :2] - T[:2, 3]
            rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
            d = np.einsum("ij,ij->i", V[:, :2], rn)
            pts.append(np.c_[P[:, :2], d, rn])
        pts = np.vstack(pts) if pts else np.zeros((0, 5))
        for b in gt["meta"]["boxes"]:
            if not b["observed"] or b["stationary"]:
                continue
            inst, j = idx_of[b["token"]]
            vs = {m: object_velocity(src, tracks, inst, j, m, b["group"], False) for m in METHODS}
            if any(v is None for v in vs.values()) or max(np.hypot(*v) for v in vs.values()) < 1.0:
                continue
            c = b["corners"]
            ctr = c.mean(0)
            grown = ctr + (c - ctr) * (1 + 0.5 / np.maximum(np.linalg.norm(c - ctr, axis=1, keepdims=True), 1e-6) * 1.0)
            inside = Path(grown).contains_points(pts[:, :2])
            if inside.sum() < 1:
                continue
            q = pts[inside]
            for m, v in vs.items():
                res[m].extend((q[:, 3:5] @ (R_g2e @ v) - q[:, 2]).tolist())
            res["null"].extend((-q[:, 2]).tolist())
            meta_rows.append((float(np.hypot(*vs["dev"])), int(inside.sum()), sc["name"]))
        tok = s["next"]
    print(sc["name"], flush=True)
print(f"\nmoving observed boxes with radar points: {len(meta_rows)}, radar points inside: {len(res['null'])}")
print("estimator | RMSE (m/s) | median |err| (m/s) | within 0.5 m/s")
for m in ["null"] + METHODS:
    e = np.abs(np.array(res[m]))
    print(f"{m:8s} | {np.sqrt((e ** 2).mean()):.3f} | {np.median(e):.3f} | {np.mean(e < 0.5):.1%}")

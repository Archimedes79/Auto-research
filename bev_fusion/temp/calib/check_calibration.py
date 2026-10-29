"""One-off (loop 2026-10-29, request "make sure calibration is right"): data-driven calibration audit on real VAL frames.

A  geometry: our chain (global -> ego pose of the camera frame -> T_cam_car -> K -> ida) against the nuScenes devkit
   (get_sample_data boxes in the camera frame, view_points) for every visible vehicle box: max difference of the camera-frame
   centre (m) and of the pixel (px). Must be ~1e-5.
B  camera + frozen depth head: probability mass of the cached depth distribution inside the box's depth interval at the
   projected box centre, as a function of an image shift (du, dv) in pixels of the 704 x 256 input. A right calibration peaks
   at (0, 0). Also the depth bias E[d] - z_centre against distance.
C  radar: number of radar detections (the dataset's own `_radar`, cached) inside the ground footprint of vehicle boxes (enlarged by
   0.5 m) as a function of a shift (dx, dy) in metres, and of the box position shifted by the box velocity x time offset.
   Peak at (0, 0) = radar extrinsics, ego motion and Doppler shift are right.
Usage (bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python temp/calib/check_calibration.py [stride, default 8]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
from feature_cache import CachedFrontRadarDataset, FeatureStore  # noqa: E402
from gt_bev import pose_matrix  # noqa: E402
from lift_pd import Grid  # noqa: E402
from camera_encoder import depth_centres  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.data_classes import Box  # noqa: E402
from nuscenes.utils.geometry_utils import view_points  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 8
root = os.path.expanduser("~/data/nuscenes_full")
nusc = NuScenes("v1.0-trainval", dataroot=root, verbose=False)
val_names = set(json.load(open("data/scenes_parts1_6.json"))["val"])
scenes = [s["token"] for s in nusc.scene if s["name"] in val_names]
store = FeatureStore("data/cache/cam_feats")
ds = CachedFrontRadarDataset(root, "v1.0-trainval", scenes, store=store, radar_cache="data/cache/radar", with_gt=False,
                             nusc=nusc)
idx = list(range(0, len(ds), stride))
print(f"{len(idx)} VAL samples of {len(ds)} ({len(scenes)} scenes)")
VEH = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer", "vehicle.construction")
dc = depth_centres()

# ---- collect the vehicle boxes of every sample in the ego frame (camera time) and the camera frame
rows = []
for n, i in enumerate(idx):
    tok = ds.samples[i]
    sample = nusc.get("sample", tok)
    cam = ds.gt.cam(tok)
    ego = nusc.get("ego_pose", cam["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    T_c2e = pose_matrix(cs)
    T_e2c = np.linalg.inv(T_c2e)
    T_g2e = np.linalg.inv(pose_matrix(ego))
    _, dev_boxes, _ = nusc.get_sample_data(cam["token"], selected_anntokens=None)
    dev = {}
    for b in dev_boxes:
        dev[b.token] = b
    boxes = []
    for at in sample["anns"]:
        a = nusc.get("sample_annotation", at)
        if not a["category_name"].startswith(VEH) or int(a["visibility_token"]) < 3:
            continue
        c_g = np.array(a["translation"])
        c_e = (T_g2e @ np.append(c_g, 1))[:3]
        c_c = (T_e2c @ np.append(c_e, 1))[:3]
        yaw_e = Quaternion(a["rotation"]).yaw_pitch_roll[0] - Quaternion(ego["rotation"]).yaw_pitch_roll[0]
        vel = nusc.box_velocity(at)                                    # global m/s (may be nan)
        boxes.append({"tok": at, "c_e": c_e, "c_c": c_c, "yaw_e": yaw_e, "size": a["size"], "vel": vel,
                      "dev": dev.get(at)})
    rows.append({"i": i, "tok": tok, "cam": cam, "K": K, "T_e2c": T_e2c, "boxes": boxes, "ego": ego})

# ---------------------------------------------------------------- A geometry
dc_max, dp_max, n_box = 0.0, 0.0, 0
ida = ds.ida.numpy()
for r in rows:
    for b in r["boxes"]:
        if b["dev"] is None:
            continue
        n_box += 1
        d = b["dev"].center
        dc_max = max(dc_max, float(np.abs(d - b["c_c"]).max()))
        uv_dev = view_points(d[:, None], r["K"], normalize=True)[:2, 0]
        p = r["K"] @ b["c_c"]
        uv_ours = p[:2] / p[2]
        dp_max = max(dp_max, float(np.abs(uv_dev - uv_ours).max()))
print(f"\nA geometry: {n_box} boxes; max |camera-frame centre ours - devkit| = {dc_max:.2e} m; max |pixel ours - devkit| = {dp_max:.2e} px")

# ---------------------------------------------------------------- B camera + depth head
shifts = list(range(-48, 49, 8))
mass = np.zeros((len(shifts), len(shifts)))
cnt = np.zeros_like(mass)
bias = {}
H, W = 256, 704
for r in rows:
    prob, _ = store.get(r["cam"]["token"])
    prob = prob.float()                                           # (118, 16, 44)
    for b in r["boxes"]:
        zc = b["c_c"][2]
        if not 10.0 < zc < 45.0:
            continue
        L, Wd, Hh = b["size"][1], b["size"][0], b["size"][2]
        half = 0.5 * max(L, Wd)
        zlo, zhi = zc - half, zc + half
        p = r["K"] @ b["c_c"]
        u0, v0 = p[:2] / p[2]
        u0, v0 = ida[0, 0] * u0 + ida[0, 2], ida[1, 1] * v0 + ida[1, 2]            # pixel of the 704 x 256 input
        if not (0 < u0 < W and 0 < v0 < H):
            continue
        m = ((dc >= zlo) & (dc <= zhi)).float()
        for a, du in enumerate(shifts):
            for c, dv in enumerate(shifts):
                u, v = u0 + du, v0 + dv
                if not (0 <= u < W and 0 <= v < H):
                    continue
                gx, gy = 2 * (u + 0.5) / W - 1, 2 * (v + 0.5) / H - 1
                pr = torch.nn.functional.grid_sample(prob[None], torch.tensor([[[[gx, gy]]]]).float(), mode="bilinear",
                                                     padding_mode="border", align_corners=False)[0, :, 0, 0]
                mass[a, c] += float((pr * m).sum())
                cnt[a, c] += 1
                if du == 0 and dv == 0:
                    ed = float((pr * dc).sum() / pr.sum().clamp(min=1e-6))
                    bias.setdefault(int(zc // 10 * 10), []).append(ed - zc)
mean_mass = mass / np.maximum(cnt, 1)
print("\nB depth-head mass inside the box depth interval (rows: du px, cols: dv px; centre = 0,0):")
print("      " + " ".join(f"{dv:6d}" for dv in shifts))
for a, du in enumerate(shifts):
    print(f"{du:5d} " + " ".join(f"{mean_mass[a, c]:6.3f}" for c in range(len(shifts))))
a0 = shifts.index(0)
bi = np.unravel_index(mean_mass.argmax(), mean_mass.shape)
print(f"peak at du={shifts[bi[0]]}, dv={shifts[bi[1]]} (value {mean_mass.max():.3f}, at 0,0 {mean_mass[a0, a0]:.3f}, n={int(cnt[a0, a0])})")
for k in sorted(bias):
    arr = np.array(bias[k])
    print(f"  depth bias E[d]-z_centre, z in [{k}, {k + 10}) m: median {np.median(arr):+.2f} m, n={len(arr)}")

# ---------------------------------------------------------------- C radar
sh = np.arange(-3.0, 3.01, 0.5)
inside = np.zeros((len(sh), len(sh)))
n_pts = 0
tshift = [-0.4, -0.2, 0.0, 0.2, 0.4]
inside_t = np.zeros(len(tshift))
for r in rows:
    item = ds[r["i"]]
    rd = item["radar"]
    x, y = rd["x"].numpy(), rd["y"].numpy()
    sel = (x > 3) & (x < 60) & (np.abs(y) < 26)
    x, y = x[sel], y[sel]
    n_pts += len(x)
    for b in r["boxes"]:
        cx, cy = b["c_e"][:2]
        if not (3 < cx < 60 and abs(cy) < 26):
            continue
        c, s = np.cos(b["yaw_e"]), np.sin(b["yaw_e"])
        L, Wd = b["size"][1] + 1.0, b["size"][0] + 1.0
        for a, dx in enumerate(sh):
            for e, dy in enumerate(sh):
                qx, qy = x - (cx + dx), y - (cy + dy)
                lx, ly = c * qx + s * qy, -s * qx + c * qy
                inside[a, e] += int(((np.abs(lx) < L / 2) & (np.abs(ly) < Wd / 2)).sum())
        vel = b["vel"]
        if np.isfinite(vel).all():
            # velocity of the box in the ego frame at camera time (rotation only)
            Re = Quaternion(r["ego"]["rotation"]).rotation_matrix
            ve = (Re.T @ np.append(vel[:2], 0.0))[:2] if len(vel) == 3 else (Re.T @ vel)[:2]
            ve = ve if np.isfinite(ve).all() else np.zeros(2)
            ages = rd["dt"].numpy()[sel]
            for k, ts in enumerate(tshift):
                qx, qy = x - (cx + ve[0] * ts), y - (cy + ve[1] * ts)
                lx, ly = c * qx + s * qy, -s * qx + c * qy
                inside_t[k] += int(((np.abs(lx) < L / 2) & (np.abs(ly) < Wd / 2)).sum())
print(f"\nC radar points (all radars, {n_pts} in range) inside enlarged vehicle footprints, shift (dx, dy) in m; rows dx, cols dy:")
print("       " + " ".join(f"{d:6.1f}" for d in sh))
for a, dx in enumerate(sh):
    print(f"{dx:6.1f} " + " ".join(f"{int(inside[a, e]):6d}" for e in range(len(sh))))
bi = np.unravel_index(inside.argmax(), inside.shape)
a0 = int(np.argmin(np.abs(sh)))
print(f"peak at dx={sh[bi[0]]:+.1f}, dy={sh[bi[1]]:+.1f} m ({int(inside.max())} points; at 0,0: {int(inside[a0, a0])})")
print("time shift of the box along its velocity (s): " + ", ".join(f"{t:+.1f}: {int(v)}" for t, v in zip(tshift, inside_t)))

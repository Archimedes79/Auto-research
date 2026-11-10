"""Probe (plan 10.2 'depth head'): is BEVFusion's camera-only image side (training/bevfusion_camera.py, Swin-T + FPN + 1 x 1 depth conv at stride 8)
wired right, and how does its depth compare with DualBEV's frozen DepthNet on the same real front images?
Check: for cells in the central third of the image below the horizon, the flat-road depth (exact ray / ground-plane intersection from the calibrated
camera, ego z = 0) against the network's expected depth; median expected depth per flat-road-depth band (cars on the road make the medians a bit
short, they hit both networks equally). nuScenes mini CAM_FRONT keyframes, CPU, 2 threads.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/probe_bevfusion_depth.py [every_nth_frame]"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "training")
from bevfusion_camera import DEPTHS, BEVFusionImageSide, bevfusion_ida, frustum_pixels, load_image  # noqa: E402
from camera_encoder import IMG_MEAN, IMG_STD, FrontCameraEncoder, camera_vector, depth_centres, nuscenes_front_ida  # noqa: E402

torch.set_num_threads(2)
step = int(sys.argv[1]) if len(sys.argv) > 1 else 6
root = os.path.expanduser("~/data/nuscenes")
from nuscenes.nuscenes import NuScenes  # noqa: E402
nusc = NuScenes("v1.0-mini", root, verbose=False)

bf = BEVFusionImageSide().eval()
print(bf.load_pretrained()["loaded"], "tensors loaded into the BEVFusion image side", flush=True)
db = FrontCameraEncoder().eval()
rep = db.load_pretrained("models/pretrained/dualbev-r50-cbgs.pth")
assert not rep["missing"] and not rep["shape_mismatch"]
DC = depth_centres()
ida_bf, ida_db = bevfusion_ida(), nuscenes_front_ida()
xs_bf, ys_bf = frustum_pixels()


def dualbev_input(path):
    img = Image.open(path).convert("RGB")
    s = ida_db[0, 0].item()
    w, h = int(round(img.width * s)), int(round(img.height * s))
    img = img.resize((w, h), Image.BILINEAR).crop((0, h - 256, 704, h))
    x = (torch.tensor(np.asarray(img), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD)
    return x.permute(2, 0, 1)


def flat_depth(u, v, K, R, t):
    """depth along the optical axis of the ground point (z = 0 in the ego frame) seen at original pixel (u, v); nan above the horizon."""
    d = np.linalg.inv(K) @ np.array([u, v, 1.0])
    dw = R @ d
    if dw[2] >= -1e-6:
        return float("nan")
    return float(-t[2] / dw[2])


BANDS = ((6, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60))
pairs = {"bevfusion": [], "dualbev": []}
samples = nusc.sample[::step]
for n, smp in enumerate(samples):
    sd = nusc.get("sample_data", smp["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])
    path = os.path.join(root, sd["filename"])
    with torch.no_grad():
        p1, _ = bf.image_features(load_image(path)[None])
        e1 = (p1[0] * DEPTHS.view(-1, 1, 1)).sum(0)                                   # (32, 88)
        Kt = torch.tensor(K, dtype=torch.float32)
        T = torch.eye(4)
        T[:3, :3], T[:3, 3] = torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
        cv = camera_vector(Kt[None], ida_db[None, :2, :2], ida_db[None, :2, 2], T[None])
        p2, _ = db.image_features(dualbev_input(path)[None], cv)
        e2 = (p2[0] * DC.view(-1, 1, 1)).sum(0)                                       # (16, 44)
    for name, e, (fh, fw), ida, pix in (("bevfusion", e1, (32, 88), ida_bf, (xs_bf, ys_bf)),
                                        ("dualbev", e2, (16, 44), ida_db, ((torch.arange(44) + 0.5) * 16 - 0.5, (torch.arange(16) + 0.5) * 16 - 0.5))):
        xs, ys = pix
        for i in range(fh):
            for j in range(fw):
                u = (float(xs[j]) - ida[0, 2].item()) / ida[0, 0].item()               # original pixel of the cell
                v = (float(ys[i]) - ida[1, 2].item()) / ida[1, 1].item()
                if not (560 <= u <= 1040):
                    continue
                f = flat_depth(u, v, K, R, t)
                if math.isfinite(f) and 6 <= f <= 60:
                    pairs[name].append((f, float(e[i, j])))
    if n % 10 == 0:
        print(f"frame {n + 1} / {len(samples)}", flush=True)
print(f"\nmedian expected depth (m) of the cells in the central third of the image, by flat-road depth band; nuScenes mini, {len(samples)} CAM_FRONT keyframes")
print(f"{'flat-road depth':>16s} | " + " | ".join(f"{k:>22s}" for k in pairs))
for lo, hi in BANDS:
    row = []
    for k in pairs:
        v = [e for f, e in pairs[k] if lo <= f < hi]
        row.append(f"{np.median(v):6.1f} (n {len(v):4d}, {np.percentile(v, 25):4.1f}-{np.percentile(v, 75):4.1f})" if v else "-")
    print(f"{lo:5d} - {hi:3d} m      | " + " | ".join(f"{r:>22s}" for r in row))
for k, v in pairs.items():
    a = np.array(v)
    print(f"{k}: correlation of expected depth with flat-road depth {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f}, "
          f"median |expected - flat| / flat {np.median(np.abs(a[:, 1] - a[:, 0]) / a[:, 0]):.2f} over {len(a)} cells")

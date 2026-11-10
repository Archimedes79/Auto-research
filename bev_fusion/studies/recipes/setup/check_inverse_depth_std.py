"""One-off: is the depth uncertainty of the pre-trained depth head closer to constant in z or in 1/z?

For the front images of a few nuScenes mini keyframes: per feature cell the mean and std of the predicted depth
distribution in z and in u = 1/z; grouped by the mean depth, the spread of std_z and std_u across depth groups shows
which coordinate makes the uncertainty homoscedastic (std about constant over range).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/setup/check_inverse_depth_std.py
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "training")
from camera_encoder import (FrontCameraEncoder, IMG_MEAN, IMG_STD, camera_vector, depth_centres,  # noqa: E402
                            nuscenes_front_ida)
from nuscenes.nuscenes import NuScenes  # noqa: E402

enc = FrontCameraEncoder()
enc.load_pretrained("models/pretrained/dualbev-r50-cbgs.pth")
enc.freeze_camera_heads()
enc.eval()
root = os.path.expanduser("~/data/nuscenes")
nusc = NuScenes("v1.0-mini", dataroot=root, verbose=False)
ida = nuscenes_front_ida()
z = depth_centres().view(-1, 1, 1)
u = 1.0 / z
rows = []
for s in nusc.sample[::8][:40]:
    sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
    s2e = torch.eye(4)
    s2e[:3, :3] = torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32)
    s2e[:3, 3] = torch.tensor(cs["translation"], dtype=torch.float32)
    img = Image.open(os.path.join(root, sd["filename"])).convert("RGB").resize((704, 396)).crop((0, 140, 704, 396))
    x = ((torch.tensor(np.asarray(img), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD))
    cam = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], s2e[None])
    with torch.no_grad():
        p = enc.image_features(x.permute(2, 0, 1)[None], cam)[0][0]
    mz = (p * z).sum(0)
    sz = ((p * (z - mz) ** 2).sum(0)).sqrt()
    mu = (p * u).sum(0)
    su = ((p * (u - mu) ** 2).sum(0)).sqrt()
    rows.append(torch.stack([mz.flatten(), sz.flatten(), su.flatten()], 1))
a = torch.cat(rows).numpy()
edges = [2, 5, 10, 15, 20, 30, 40, 60]
print("mean depth group | cells | median std_z (m) | median std_u (1/m) | std_u * z^2 (m)")
med_z, med_u = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if m.sum() < 20:
        continue
    zc = np.median(a[m, 0])
    sz, su = np.median(a[m, 1]), np.median(a[m, 2])
    med_z.append(sz)
    med_u.append(su)
    print(f"{lo:4.0f}-{hi:<4.0f} m      | {m.sum():6d} | {sz:8.2f}        | {su:.4f}            | {su * zc * zc:6.2f}")
rz, ru = max(med_z) / min(med_z), max(med_u) / min(med_u)
print(f"max/min over the groups: std_z {rz:.1f}x, std_u {ru:.1f}x  ->  "
      f"{'closer to constant in 1/z' if ru < rz else 'closer to constant in z'}")

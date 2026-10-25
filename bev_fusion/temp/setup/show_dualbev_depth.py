"""One-off: picture of the loaded front-camera encoder on one nuScenes CAM_FRONT keyframe.

Panels: the network input (256 x 704), the expected depth per feature cell, the depth uncertainty (std of the depth
distribution), and the BEV (norm of the lifted features, x forward up, y left left).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/setup/show_dualbev_depth.py <out.png>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

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
sdata = nusc.get("sample_data", nusc.sample[10]["data"]["CAM_FRONT"])
cs = nusc.get("calibrated_sensor", sdata["calibrated_sensor_token"])
K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
s2e = torch.eye(4)
s2e[:3, :3] = torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32)
s2e[:3, 3] = torch.tensor(cs["translation"], dtype=torch.float32)
ida = nuscenes_front_ida()
img = Image.open(os.path.join(root, sdata["filename"])).convert("RGB").resize((704, 396), Image.BILINEAR)
img = img.crop((0, 140, 704, 396))
x = ((torch.tensor(np.asarray(img), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD))
x = x.permute(2, 0, 1)[None]
cam = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], s2e[None])
with torch.no_grad():
    prob, ctx = enc.image_features(x, cam)
    vol = enc.lift(ctx, prob, K, torch.linalg.inv(s2e), (256, 704), ida)
dc = depth_centres().view(-1, 1, 1)
mean = (prob[0] * dc).sum(0)
std = ((prob[0] * (dc - mean) ** 2).sum(0)).sqrt()
bev = vol[0].norm(dim=0)                                         # (120, 104): x forward, y left

fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), gridspec_kw={"width_ratios": [2.75, 2.75, 2.75, 1]})
ax[0].imshow(img)
ax[0].set_title("input: CAM_FRONT, 256 x 704")
im1 = ax[1].imshow(mean, cmap="viridis", vmin=0, vmax=60)
ax[1].set_title("expected depth per feature cell (m)")
plt.colorbar(im1, ax=ax[1], fraction=0.025)
im2 = ax[2].imshow(std, cmap="magma")
ax[2].set_title("depth uncertainty: std of the distribution (m)")
plt.colorbar(im2, ax=ax[2], fraction=0.025)
ax[3].imshow(bev.flip(0).flip(1), cmap="gray", extent=[26, -26, 0, 60])
ax[3].set_title("BEV: lifted feature norm")
ax[3].set_xlabel("y (m, left positive)")
ax[3].set_ylabel("x forward (m)")
for a in ax[:3]:
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("front-camera encoder with pre-trained DualBEV heads (frozen), lift PD - nuScenes mini, one keyframe")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
print("saved", sys.argv[1])

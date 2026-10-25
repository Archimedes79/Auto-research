"""One-off: does FrontCameraEncoder load the DualBEV checkpoint, and do the loaded heads see depth in a real front image?

1. keys of the checkpoint per prefix, our load report (loaded / missing / unexpected / shape mismatch)
2. functional check on one nuScenes CAM_FRONT keyframe: the expected depth per feature cell must grow from the bottom
   rows (road near the car) to the rows near the horizon - random weights do not do that
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/setup/check_dualbev_load.py
"""
import collections
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "training")
from camera_encoder import (FrontCameraEncoder, IMG_MEAN, IMG_STD, camera_vector, depth_centres,  # noqa: E402
                            nuscenes_front_ida)

ck_path = "models/pretrained/dualbev-r50-cbgs.pth"
ck = torch.load(ck_path, map_location="cpu", weights_only=True)
print("top-level keys:", list(ck)[:6])
sd = ck.get("state_dict", ck)
pre = collections.Counter(k.split(".")[0] + ("." + k.split(".")[1] if k.startswith("img_view") else "") for k in sd)
print("keys per prefix:", dict(pre))

enc = FrontCameraEncoder()
rep = enc.load_pretrained(ck_path)
print(f"loaded {rep['loaded']}, missing {len(rep['missing'])}, unexpected {len(rep['unexpected'])}, "
      f"shape mismatch {len(rep['shape_mismatch'])}")
for name in ("missing", "unexpected", "shape_mismatch"):
    for x in rep[name][:8]:
        print(f"  {name}: {x}")
enc.freeze_camera_heads()
enc.eval()

# one real front image with its calibration (nuScenes mini, read through the WSL view)
from nuscenes.nuscenes import NuScenes  # noqa: E402

nusc = NuScenes("v1.0-mini", dataroot=os.path.expanduser("~/data/nuscenes"), verbose=False)
sdata = nusc.get("sample_data", nusc.sample[10]["data"]["CAM_FRONT"])
cs = nusc.get("calibrated_sensor", sdata["calibrated_sensor_token"])
from pyquaternion import Quaternion  # noqa: E402

K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
s2e = torch.eye(4)
s2e[:3, :3] = torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32)
s2e[:3, 3] = torch.tensor(cs["translation"], dtype=torch.float32)
ida = nuscenes_front_ida()
img = Image.open(os.path.join(os.path.expanduser("~/data/nuscenes"), sdata["filename"])).convert("RGB")
img = img.resize((704, 396), Image.BILINEAR).crop((0, 140, 704, 396))
x = (torch.tensor(np.asarray(img), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD)
x = x.permute(2, 0, 1)[None]
cam = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], s2e[None])
with torch.no_grad():
    prob, ctx = enc.image_features(x, cam)
    bev = enc(x, K, torch.linalg.inv(s2e), ida, cam)
exp_depth = (prob[0] * depth_centres().view(-1, 1, 1)).sum(0)            # (16, 44)
peak = prob[0].max(0).values
rows = exp_depth.mean(1)
print("expected depth per feature row (top -> bottom, m):", " ".join(f"{v:.1f}" for v in rows.tolist()))
print(f"mean peak probability {peak.mean():.3f} (uniform would be {1 / prob.shape[1]:.4f})")
print(f"BEV {tuple(bev.shape)}, non-zero share {(bev.abs() > 1e-6).float().mean():.3f}")
ok = rep["loaded"] > 0 and not rep["shape_mismatch"] and not rep["missing"] and rows[-1] < rows[6] and peak.mean() > 5 / 118
print("LOAD CHECK PASS" if ok else "LOAD CHECK FAIL")

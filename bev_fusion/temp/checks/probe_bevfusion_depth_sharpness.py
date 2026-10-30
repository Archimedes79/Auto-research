"""How sharp is BEVFusion's depth distribution? peak probability, entropy against uniform (ln 118), the spread of expected depth, mean expected
depth by image row; same numbers for DualBEV. nuScenes mini, 12 CAM_FRONT keyframes, CPU 2 threads.
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/probe_bevfusion_depth_sharpness.py"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "training")
from bevfusion_camera import DEPTHS, BEVFusionImageSide, load_image  # noqa: E402
from camera_encoder import IMG_MEAN, IMG_STD, FrontCameraEncoder, camera_vector, depth_centres, nuscenes_front_ida  # noqa: E402

torch.set_num_threads(2)
root = os.path.expanduser("~/data/nuscenes")
from nuscenes.nuscenes import NuScenes  # noqa: E402
nusc = NuScenes("v1.0-mini", root, verbose=False)
bf = BEVFusionImageSide().eval()
bf.load_pretrained()
db = FrontCameraEncoder().eval()
db.load_pretrained("models/pretrained/dualbev-r50-cbgs.pth")
ida = nuscenes_front_ida()
stats = {"bevfusion": [], "dualbev": []}
rows = {"bevfusion": [], "dualbev": []}
for smp in nusc.sample[::34][:12]:
    sd = nusc.get("sample_data", smp["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    path = os.path.join(root, sd["filename"])
    img = Image.open(path).convert("RGB")
    s = ida[0, 0].item()
    w, h = int(round(img.width * s)), int(round(img.height * s))
    x = (torch.tensor(np.asarray(img.resize((w, h), Image.BILINEAR).crop((0, h - 256, 704, h))), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD)
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
    from gt_bev import pose_matrix  # noqa: E402
    T = torch.tensor(pose_matrix(cs), dtype=torch.float32)
    cv = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], T[None])
    with torch.no_grad():
        for name, (p, dc) in (("bevfusion", (bf.image_features(load_image(path)[None])[0], DEPTHS)),
                              ("dualbev", (db.image_features(x.permute(2, 0, 1)[None] if False else x.permute(2, 0, 1)[None], cv)[0], depth_centres()))):
            p = p[0]
            ent = -(p * p.clamp(min=1e-12).log()).sum(0) / math.log(118)
            ed = (p * dc.view(-1, 1, 1)).sum(0)
            stats[name].append((float(p.max(0).values.mean()), float(ent.mean()), float(ed.std())))
            rows[name].append(ed.mean(1).numpy())
for name in stats:
    a = np.mean(stats[name], 0)
    r = np.mean(rows[name], 0)
    pick = np.linspace(0, len(r) - 1, 8).astype(int)
    print(f"{name:9s} mean peak probability {a[0]:.3f} (uniform {1 / 118:.3f}), mean entropy / ln 118 = {a[1]:.2f} (uniform 1.00), "
          f"std of expected depth over cells {a[2]:.1f} m; mean expected depth by feature row {[round(float(r[i]), 1) for i in pick]}")

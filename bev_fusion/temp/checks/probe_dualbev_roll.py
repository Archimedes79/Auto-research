"""R-GEO-2 open point (2): does the frozen DualBEV image side cope with a ROLLED camera? A pure roll about the optical axis turns the image in the
image plane about the principal point; so real front images are rotated by rho about (cx, cy) and the calibration is updated (camera-to-ego
rotation R' = R Rz(rho), same K): the physical situation of a camera mounted rolled by rho. The expected depth of the rolled image against the flat-road
depth of the rolled calibration (same probe as probe_bevfusion_depth.py: central third of the image below the horizon, nuScenes mini CAM_FRONT).
Variants: calibration updated in the 27-vector (correct) and left stale (the head is told the original rotation).
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/probe_dualbev_roll.py [every_nth_frame]"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "training")
from camera_encoder import IMG_MEAN, IMG_STD, FrontCameraEncoder, camera_vector, depth_centres, nuscenes_front_ida  # noqa: E402
from gt_bev import pose_matrix  # noqa: E402

torch.set_num_threads(2)
step = int(sys.argv[1]) if len(sys.argv) > 1 else 12
root = os.path.expanduser("~/data/nuscenes")
from nuscenes.nuscenes import NuScenes  # noqa: E402
nusc = NuScenes("v1.0-mini", root, verbose=False)
db = FrontCameraEncoder().eval()
rep = db.load_pretrained("models/pretrained/dualbev-r50-cbgs.pth")
assert not rep["missing"] and not rep["shape_mismatch"]
DC = depth_centres()
ida = nuscenes_front_ida()


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rolled_image(img, rho, cx, cy):
    """content rotated in the image plane so that (u, v) = R2(rho) (u' - c) + c  (PIL AFFINE: output pixel -> input pixel)."""
    c, s = math.cos(rho), math.sin(rho)
    data = (c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy)
    return img.transform(img.size, Image.AFFINE, data, resample=Image.BICUBIC)


def net_input(img):
    sc = ida[0, 0].item()
    w, h = int(round(img.width * sc)), int(round(img.height * sc))
    x = (torch.tensor(np.asarray(img.resize((w, h), Image.BILINEAR).crop((0, h - 256, 704, h))), dtype=torch.float32) - torch.tensor(IMG_MEAN)) / torch.tensor(IMG_STD)
    return x.permute(2, 0, 1)


def flat_depth(u, v, K, R, t):
    d = np.linalg.inv(K) @ np.array([u, v, 1.0])
    dw = R @ d
    return float("nan") if dw[2] >= -1e-6 else float(-t[2] / dw[2])


RHOS = [0.0, 10.0, 20.0, -20.0]
res = {(r, m): [] for r in RHOS for m in ("updated", "stale")}
frames = nusc.sample[::step]
checked = False
for n, smp in enumerate(frames):
    sd = nusc.get("sample_data", smp["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])
    img0 = Image.open(os.path.join(root, sd["filename"])).convert("RGB")
    cx, cy = K[0, 2], K[1, 2]
    for rho_deg in RHOS:
        rho = math.radians(rho_deg)
        Rr = R @ rz(rho)                                                     # camera-to-ego of the rolled camera
        img = rolled_image(img0, rho, cx, cy) if rho_deg else img0
        modes = ("updated", "stale") if rho_deg in (10.0, -20.0) else ("updated",)
        for mode in modes:
            Rv = Rr if mode == "updated" else R
            T = torch.eye(4)
            T[:3, :3], T[:3, 3] = torch.tensor(Rv, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
            cv = camera_vector(torch.tensor(K, dtype=torch.float32)[None], ida[None, :2, :2], ida[None, :2, 2], T[None])
            with torch.no_grad():
                p, _ = db.image_features(net_input(img)[None], cv)
            e = (p[0] * DC.view(-1, 1, 1)).sum(0)
            for i in range(16):
                for j in range(44):
                    u = ((j + 0.5) * 16 - 0.5 - ida[0, 2].item()) / ida[0, 0].item()      # pixel of the rolled image
                    v = ((i + 0.5) * 16 - 0.5 - ida[1, 2].item()) / ida[1, 1].item()
                    if not (560 <= u <= 1040):
                        continue
                    # the source pixel of the unrolled image must exist (rotation fills the corners with black)
                    c, s = math.cos(rho), math.sin(rho)
                    us, vs = c * (u - cx) - s * (v - cy) + cx, s * (u - cx) + c * (v - cy) + cy
                    if not (0 <= us < 1600 and 0 <= vs < 900):
                        continue
                    f = flat_depth(u, v, K, Rr, t)                                 # flat-road depth of the ROLLED camera at its pixel
                    if math.isfinite(f) and 6 <= f <= 60:
                        res[(rho_deg, mode)].append((f, float(e[i, j])))
                        if not checked and rho_deg:                                 # derivation check: same ground point as in the unrolled image
                            f0 = flat_depth(us, vs, K, R, t)
                            assert abs(f0 - f) < 1e-6 * max(1, f), (f0, f)
                            checked = True
    if n % 6 == 0:
        print(f"frame {n + 1} / {len(frames)}", flush=True)
print(f"\nDualBEV frozen image side, rolled camera (real front images rotated in the image plane + calibration updated); nuScenes mini, {len(frames)} keyframes")
print("roll (deg) | 27-vector | cells | corr. with flat-road depth | median |expected - flat| / flat | median expected depth at flat 6-10 / 15-20 / 30-45 m")
for (rho, mode), v in res.items():
    if not v:
        continue
    a = np.array(v)
    band = lambda lo, hi: np.median(a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1])      # noqa: E731
    print(f"{rho:9.0f}  | {mode:8s}  | {len(a):5d} | {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f} | {np.median(np.abs(a[:, 1] - a[:, 0]) / a[:, 0]):.2f} | "
          f"{band(6, 10):.1f} / {band(15, 20):.1f} / {band(30, 45):.1f}")

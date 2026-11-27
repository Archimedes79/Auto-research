"""A synthetic cached batch for the tiny model (CachedBEVFusionModel on a small grid): front-camera features of the kind the config names, five radars as on
the car, ego motion. `batch_for` is the one of v1 tests/test_slots.py (the slot contract test, not kept in v2), copied unchanged.
    from synthetic_batch import batch_for          # tests/support and the model folders net/{prep,model,loss} on sys.path
    batch = batch_for(model.cfg)                   # B 2, 3 frames, seed 0"""
import math

import torch

from bevfusion_camera import DEPTHS, bevfusion_ida
from camera_encoder import nuscenes_front_ida
from feature_cache import FEAT_HW, N_CTX, N_PROB
from interface import RADAR_FIELDS
from lift_pd import _front_camera
from nuscenes_dataset import DST_HW


def batch_for(cfg, B=2, Fn=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, T, _ = _front_camera(torch.float32)
    K = K.clone()
    K[:2] *= 2                                                   # the synthetic camera at the original 1600 x 900 resolution
    bevf = cfg.camera_features == "bevfusion"                    # the cached BEVFusion features: own depth bins, stride 8, own crop
    n_d, hw_f = (len(DEPTHS), (32, 88)) if bevf else (N_PROB, FEAT_HW)
    prob = torch.rand(B, Fn, n_d, *hw_f, generator=g)
    rig = torch.eye(4).repeat(5, 1, 1)                           # five radars as on the car: front, two front corners, two rear
    for i, (x, y, yaw) in enumerate([(3.4, 0.0, 0.0), (2.4, 0.8, 1.04), (2.4, -0.8, -1.04), (-1.0, 0.9, 2.1), (-1.0, -0.9, -2.1)]):
        rig[i, :2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rig[i, :2, 3] = torch.tensor([x, y])
    b = {"K": K.expand(B, Fn, 3, 3).clone(), "T_cam_car": T.expand(B, Fn, 4, 4).clone(), "radar_poses": rig.expand(B, 5, 4, 4).clone(),
         "ida": bevfusion_ida() if bevf else nuscenes_front_ida(dst_hw=DST_HW),
         "hw": DST_HW, "prob": prob / prob.sum(2, keepdim=True), "ctx": torch.randn(B, Fn, N_CTX, *hw_f, generator=g),
         "v": torch.full((B,), 8.0), "omega": torch.full((B,), 0.05),
         "steps": [[]] + [[(torch.full((B,), 1 / 12), torch.full((B,), 8.0), torch.full((B,), 0.05))] * k for k in range(1, Fn)],
         "radar": dict({f: torch.randn(40, generator=g) for f in RADAR_FIELDS},          # the contract's detection fields
                       x=torch.rand(40, generator=g) * 15, y=torch.rand(40, generator=g) * 14 - 7, r=torch.rand(40, generator=g) * 15,
                       dt=torch.rand(40, generator=g) * 0.5, b=torch.arange(40) % B)}
    if cfg.camera_features == "dual":
        p2 = torch.rand(B, Fn, len(DEPTHS), 32, 88, generator=g)
        b.update(prob2=p2 / p2.sum(2, keepdim=True), ctx2=torch.randn(B, Fn, N_CTX, 32, 88, generator=g), ida2=bevfusion_ida())
    if cfg.camera_student:                                       # t22: the student reads the current key-frame image
        b["lane_img"] = torch.randint(0, 256, (B, 3, *DST_HW), dtype=torch.uint8, generator=g)
    return b

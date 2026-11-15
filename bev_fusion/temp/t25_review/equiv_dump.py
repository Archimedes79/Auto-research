"""dump outputs / state dict keys / grads of the default-path configs for a given training dir: python equiv_dump.py <training_dir> <out.pt>"""
import math
import os
import sys

import torch

TR, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, TR)
sys.path.insert(1, "/mnt/c/Auto-research")
torch.set_num_threads(2)
from bevfusion_camera import DEPTHS, bevfusion_ida  # noqa: E402
from camera_encoder import nuscenes_front_ida  # noqa: E402
from feature_cache import FEAT_HW, N_CTX, N_PROB, CachedBEVFusionModel  # noqa: E402
from interface import RADAR_FIELDS  # noqa: E402
from lift_pd import _front_camera  # noqa: E402
from model import ModelConfig  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402
import model as M  # noqa: E402
print("model from", M.__file__)

GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
BASE = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0, dense_outputs=True)


def batch_for(cfg, B=2, Fn=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, T, _ = _front_camera(torch.float32)
    K = K.clone()
    K[:2] *= 2
    bevf = cfg.camera_features == "bevfusion"
    n_d, hw_f = (len(DEPTHS), (32, 88)) if bevf else (N_PROB, FEAT_HW)
    prob = torch.rand(B, Fn, n_d, *hw_f, generator=g)
    rig = torch.eye(4).repeat(5, 1, 1)
    for i, (x, y, yaw) in enumerate([(3.4, 0.0, 0.0), (2.4, 0.8, 1.04), (2.4, -0.8, -1.04), (-1.0, 0.9, 2.1), (-1.0, -0.9, -2.1)]):
        rig[i, :2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rig[i, :2, 3] = torch.tensor([x, y])
    b = {"K": K.expand(B, Fn, 3, 3).clone(), "T_cam_car": T.expand(B, Fn, 4, 4).clone(), "radar_poses": rig.expand(B, 5, 4, 4).clone(),
         "ida": bevfusion_ida() if bevf else nuscenes_front_ida(dst_hw=DST_HW),
         "hw": DST_HW, "prob": prob / prob.sum(2, keepdim=True), "ctx": torch.randn(B, Fn, N_CTX, *hw_f, generator=g),
         "v": torch.full((B,), 8.0), "omega": torch.full((B,), 0.05),
         "steps": [[]] + [[(torch.full((B,), 1 / 12), torch.full((B,), 8.0), torch.full((B,), 0.05))] * k for k in range(1, Fn)],
         "radar": dict({f: torch.randn(40, generator=g) for f in RADAR_FIELDS},
                       x=torch.rand(40, generator=g) * 15, y=torch.rand(40, generator=g) * 14 - 7, r=torch.rand(40, generator=g) * 15,
                       dt=torch.rand(40, generator=g) * 0.5, b=torch.arange(40) % B)}
    if cfg.camera_features == "dual":
        p2 = torch.rand(B, Fn, len(DEPTHS), 32, 88, generator=g)
        b.update(prob2=p2 / p2.sum(2, keepdim=True), ctx2=torch.randn(B, Fn, N_CTX, 32, 88, generator=g), ida2=bevfusion_ida())
    return b


def flat(res):
    out = {"features": res["features"].detach().clone()}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    out[f"{dt}.{k}.{kk}"] = t.detach().clone()

    def walk(prefix, x):
        if torch.is_tensor(x):
            out[prefix] = x.detach().clone()
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(prefix + "." + str(k), v)
    walk("state", res["state"])
    side = res.get("radar_side")
    if side:
        out["side.features"], out["side.state"] = side["features"].detach().clone(), side["state"].detach().clone()
    return out


CONFIGS = {
    "window": dict(),
    "window_mem_cam": dict(cam_temporal="window_memory"),
    "window_rad_mem": dict(cam_temporal="window_memory", rad_temporal="window_memory", rad_static_gate=True, radar_kind="delta_slots", c_rad=64),
    "window_rad_hits": dict(cam_temporal="window_memory", rad_temporal="window_memory"),
    "window_rad_ego_gru": dict(cam_temporal="window_memory", rad_temporal="ego_gru"),
    "pair": dict(cam_temporal="window_memory", cam_buffer="pair", motion_block=True, box_vel_from_field=True),
    "recurrent": dict(cam_temporal="window_memory", cam_buffer="recurrent", motion_block=True, box_vel_from_field=True, radar_kind="delta_slots", c_rad=64),
    "recurrent_radmem": dict(cam_temporal="window_memory", cam_buffer="recurrent", motion_block=True, box_vel_from_field=True, radar_kind="delta_slots", c_rad=64,
                             rad_temporal="window_memory", rad_static_gate=True),
    "raster": dict(radar_kind="raster"),
    "radar_off": dict(radar_kind="off"),
}
res = {}
for name, kw in CONFIGS.items():
    cfg = ModelConfig(**dict(BASE, **kw))
    torch.manual_seed(0)
    m = CachedBEVFusionModel(cfg)
    with torch.no_grad():
        for n, q in m.named_parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 0.05)
    keys = {k: (tuple(v.shape), float(v.double().sum())) for k, v in m.state_dict().items()}
    entry = {"keys": keys, "n_param": sum(p.numel() for p in m.parameters())}
    nf = 2 if kw.get("cam_buffer") == "pair" else 3
    batch = batch_for(cfg, Fn=nf)
    for mode in ("eval", "train"):
        getattr(m, mode)()
        torch.manual_seed(1)
        m.zero_grad()
        ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
        with ctx:
            r1 = m(batch, dts=(0.0, 0.5))
            step = (torch.full((2,), 0.5), torch.full((2,), 8.0), torch.full((2,), 0.05))
            r2 = m(batch, dts=(0.0, 0.5), state=r1["state"], step=step)
        entry[mode] = (flat(r1), flat(r2))
        if mode == "train":
            loss = r2["features"].square().mean() + r2["out"][0.0]["sem"].mean() + r1["features"].square().mean()
            if r2.get("radar_side"):
                loss = loss + r2["radar_side"]["features"].square().mean()
            loss.backward()
            entry["grads"] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    if kw.get("cam_buffer") == "recurrent" and not kw.get("rad_temporal"):
        m.eval()
        with torch.no_grad():
            b1 = {k: (v[:, :1] if k in ("K", "T_cam_car", "prob", "ctx", "prob2", "ctx2") else v) for k, v in batch.items()}
            b1["steps"] = [[]]
            st = m.advance(b1)
            st = m.advance(b1, st, [(torch.full((2,), 1 / 12), torch.full((2,), 8.0), torch.full((2,), 0.05))])
        entry["advance"] = {k: v.detach().clone() for k, v in {**{"cam." + k: v for k, v in st["cam"].items()}, **{"buf." + k: v for k, v in st["buf"].items()}}.items()}
    res[name] = entry
    print(name, "ok", entry["n_param"], flush=True)
torch.save(res, OUT)

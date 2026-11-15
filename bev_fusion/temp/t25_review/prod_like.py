"""A2 with the production-like slots (delta_slots c_rad 64, motion block, box_vel_from_field, static gate): step-0 identity incl. radar_side, stream = batch, side loss path"""
import os
import sys

import torch

ROOT = "/mnt/c/Auto-research/bev_fusion"
for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch.set_num_threads(2)
from feature_cache import CachedBEVFusionModel  # noqa: E402
from interface import RADAR_FIELDS, check_inputs, check_outputs  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import batch_for  # noqa: E402

GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=64, c_fuse=24, trunk_width=0.25, p_drop=0.0,
            radar_kind="delta_slots", motion_block=True, box_vel_from_field=True, object_direction=True, object_moving=True, camera_features="dual", centreline_1m=True)
MEM = dict(cam_buffer="recurrent", rad_temporal="window_memory", rad_static_gate=True, radar_per_frame=True)
B = 2


def sweep(i, n=48):
    g = torch.Generator().manual_seed(100 + i)
    d = {f: torch.randn(n, generator=g) for f in RADAR_FIELDS}
    d.update(x=torch.rand(n, generator=g) * 15.5, y=torch.rand(n, generator=g) * 15.5 - 7.75, dt=torch.rand(n, generator=g) * 0.09, b=torch.arange(n) % B)
    d["r"] = d["x"].abs() + 1.0
    d["psi"] = torch.rand(n, generator=g) - 0.5
    d["sensor"] = (torch.arange(n) % 5).long()
    d["vr"] = torch.randn(n, generator=g) * 0.7
    return d


base_cfg = ModelConfig(**dict(TINY, **MEM))
base = batch_for(base_cfg, B=B, Fn=3)
RAD = [sweep(i) for i in range(3)]
base["radar"] = RAD[0]
base["radar_seq"] = RAD
print("check_inputs:", check_inputs(base))


def build(zero_mix, **kw):
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw))).eval()
    with torch.no_grad():
        for n, q in m.named_parameters():
            if q.requires_grad and q.abs().max() == 0 and not (zero_mix and n.startswith("rad_mix")):
                q.normal_(0, 0.05)
    return m


mem, ref = build(True, **MEM), build(True, cam_buffer="recurrent")
ref.load_state_dict({k: v for k, v in mem.state_dict().items() if k in ref.state_dict()}, strict=False)
with torch.no_grad():
    rm, rr = mem(base, dts=(0.0, 0.5)), ref({k: v for k, v in base.items() if k != "radar_seq"}, dts=(0.0, 0.5))
check_outputs(rm, (0.0, 0.5), raise_on=True)
ok = torch.equal(rm["features"], rr["features"])
for dt in rm["out"]:
    for k, v in rm["out"][dt].items():
        if torch.is_tensor(v):
            ok &= torch.equal(v, rr["out"][dt][k])
        elif isinstance(v, dict):
            ok &= all(torch.equal(t, rr["out"][dt][k][kk]) for kk, t in v.items())
print("step-0 identity (production-like slots), outputs bit-equal:", ok)
print("radar_side features / state equal:", torch.equal(rm["radar_side"]["features"], rr["radar_side"]["features"]), torch.equal(rm["radar_side"]["state"], rr["radar_side"]["state"]))
print("state keys", sorted(rm["state"]), sorted(rm["state"]["buf"]))

# side loss path as train.forward_loss uses it
enc = mem.radar
mem.train()
r = mem(base, dts=(0.0,))
la = enc.aux_loss(r["radar_side"]["features"].float(), r["radar_side"]["state"].float())
print("aux_loss finite:", bool(torch.isfinite(la)), float(la))
(r["features"].square().mean() + la).backward()
print("radar encoder grad finite & nonzero:", all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters() if p.requires_grad))
mem.eval()
# stream = batch with delta_slots / motion block
def mk(idx, order_first=True):
    b = dict(base)
    return b

def one_frame(frame_k):
    b = dict(base)
    for k in ("K", "T_cam_car", "prob", "ctx", "prob2", "ctx2"):
        b[k] = base[k][:, frame_k:frame_k + 1]
    b["steps"] = [base["steps"][frame_k]]
    b["radar_seq"] = [base["radar_seq"][frame_k]]
    b["radar"] = base["radar_seq"][frame_k]
    return b
# base["steps"][k] = chain from frame k to the newest; frames are newest first: 2 (oldest) -> 1 -> 0
with torch.no_grad():
    full = mem(base, dts=(0.0,))
    st = None
    for k in (2, 1, 0):
        step = None
        if k < 2:
            n = len(base["steps"][k + 1]) - len(base["steps"][k])
            step = base["steps"][k + 1][:n]
        res = mem(one_frame(k), dts=(0.0,), state=st, step=step)
        st = res["state"]
gap = (res["features"] - full["features"]).abs().max().item()
print("stream (3 one-frame steps) vs batch features gap:", gap)
for kk in ("M", "W", "pose"):
    print("rad state", kk, (st["rad"][kk] - full["state"]["rad"][kk]).abs().max().item())
print("rad_fused gap", (st["buf"]["rad_fused"] - full["state"]["buf"]["rad_fused"]).abs().max().item())

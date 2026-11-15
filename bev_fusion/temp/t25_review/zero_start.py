import os
import sys

import torch

ROOT = "/mnt/c/Auto-research/bev_fusion"
for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch.set_num_threads(2)
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import batch_for  # noqa: E402

GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0)
cfg = ModelConfig(**dict(TINY, cam_buffer="recurrent", rad_temporal="window_memory", rad_static_gate=True, radar_per_frame=True))
torch.manual_seed(0)
m = CachedBEVFusionModel(cfg).train()
with torch.no_grad():                       # zero-start layers of the OTHER modules get values (as the warm start has), rad_mix stays zero
    for n, q in m.named_parameters():
        if q.requires_grad and q.abs().max() == 0 and not n.startswith("rad_mix"):
            q.normal_(0, 0.05)
b = batch_for(cfg, Fn=3)
b["radar_seq"] = [b["radar"]] * 3
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4 / 3, weight_decay=0.01, eps=1e-8)
for it in range(4):
    m.zero_grad()
    r = m(b, dts=(0.0,))
    loss = r["features"].square().mean() + r["out"][0.0]["sem"].square().mean()
    loss.backward()
    blk = [p.grad.abs().max().item() for n, p in m.named_parameters() if n.startswith("rad_temporal.") and p.grad is not None]
    rad = [p.grad.abs().max().item() for n, p in m.named_parameters() if n.startswith("radar.") and p.grad is not None]
    mix = m.rad_mix.weight.grad.abs().max().item()
    print(f"step {it}: |rad_mix.w|max {m.rad_mix.weight.abs().max().item():.2e}  rad_mix grad max {mix:.2e}  block grad max {max(blk):.2e}  radar-encoder grad max {max(rad):.2e}  none-grad block params {sum(1 for n, p in m.named_parameters() if n.startswith('rad_temporal.') and p.grad is None)}")
    opt.step()

"""The radar slot "delta_slots" (training/radar_encoder.py::RadarEncoder) inside the whole chain: wiring, side loss, monitor, old path.

Real nuScenes mini (2 mid-scene keyframes, GT of k-1 / k / k+1), CPU, camera off (the radar branch alone carries the BEV), width 0.5:
  1. check_config refuses c_rad < 61 for delta_slots (the pass-through start needs one channel per input channel)
  2. the model returns 'radar_side' = encoder features + lossless raster, shapes (B, c_rad, X, Y) / (B, 61, X, Y); the output contract passes
  3. forward_loss with radar_aux > 0 adds weight x side loss to the total (weight = radar_encoder.aux_weight(step_frac, ...)), logs
     'radar_aux'; the gradient reaches the radar convs, the side-loss decoder AND the trunk; with radar_aux = 0 nothing is added and the
     decoder gets no gradient
  4. validation (train_terms=False): 'radar_kept' in [0, 1], no side loss in the total
  5. the old slot is untouched: radar_kind 'raster' returns no 'radar_side' and radar_aux has no effect on its loss
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_radar_slot.py
"""
import os
import sys

import torch

sys.path.insert(0, "training")
from interface import check_outputs  # noqa: E402
from losses import BEVLoss, sample_targets  # noqa: E402
from model import BEVFusionModel, ModelConfig, check_config  # noqa: E402
from nuscenes_dataset import FrontRadarDataset, collate  # noqa: E402
from radar_encoder import aux_weight, n_channels  # noqa: E402
from train import TrainConfig, forward_loss  # noqa: E402

torch.manual_seed(0)
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def build(**kw):
    torch.manual_seed(0)
    return BEVFusionModel(ModelConfig(pretrained="", camera_off=True, trunk_width=0.5, **kw))


ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
batch = collate([ds[10], ds[200]])
dts = tuple(sorted(sample_targets(batch["gt"])))

# 1 config check
try:
    check_config(ModelConfig(pretrained="", radar_kind="delta_slots", c_rad=32))
    check(False, "delta_slots with c_rad = 32 is refused")
except ValueError as e:
    check("c_rad" in str(e) and str(n_channels()) in str(e), f"delta_slots with c_rad = 32 is refused ({str(e)[:70]}...)")

# 2 side outputs and the output contract
m = build(radar_kind="delta_slots", c_rad=64)
m.train()
res = m(batch, dts=dts)
side = res.get("radar_side")
X, Y = m.grid.nx, m.grid.ny
check(side is not None and side["features"].shape == (2, 64, X, Y) and side["state"].shape == (2, n_channels(), X, Y),
      "radar_side = encoder features (2, 64, X, Y) + lossless raster (2, 61, X, Y)")
check(check_outputs(res, dts) == [], "the output contract passes with the extra key")

# 3 side loss in the total, gradients
loss = BEVLoss(grid=m.grid)
cfg = TrainConfig(radar_aux=1.0, radar_aux_floor=0.05, radar_aux_half=0.2)
m.zero_grad()
out, _, _ = forward_loss(m, loss, batch, 0.5, "cpu", "off", cfg)
w = aux_weight(0.5, 1.0, 0.05, 0.2)
check("radar_aux" in out and abs(float(out["total"]) - float(out["total_main"]) - w * float(out["radar_aux"])) < 1e-4 * max(1.0, float(out["total"])),
      f"total = main + {w:.3f} x side loss ({float(out['radar_aux']):.3f})")
out["total"].backward()
g = lambda p: p.grad is not None and bool(p.grad.abs().sum() > 0)
check(g(m.radar.net[0].weight) and g(m.radar.decoder[0].weight) and any(g(q) for q in m.trunk.parameters()),
      "the gradient reaches the encoder convs, the side-loss decoder and the downstream blocks")
m.zero_grad()
out0, _, _ = forward_loss(m, loss, batch, 0.5, "cpu", "off", TrainConfig(radar_aux=0.0))
out0["total"].backward()
check("radar_aux" not in out0 and float(out0["total"]) == float(out0["total_main"]) and not g(m.radar.decoder[0].weight),
      "radar_aux = 0: nothing added, the decoder gets no gradient")

# 4 validation: monitor only
m.eval()
with torch.no_grad():
    ov, _, _ = forward_loss(m, loss, batch, 0.5, "cpu", "off", cfg, train_terms=False)
check("radar_kept" in ov and 0.0 <= float(ov["radar_kept"]) <= 1.0 and "radar_aux" not in ov and float(ov["total"]) == float(ov["total_main"]),
      f"validation logs radar_kept ({float(ov['radar_kept']):.3f}) and adds no side loss")

# 5 the old slot is untouched
r = build(radar_kind="raster")
r.eval()
with torch.no_grad():
    a, _, _ = forward_loss(r, BEVLoss(grid=r.grid), batch, 0.5, "cpu", "off", TrainConfig(radar_aux=0.0))
    b, _, _ = forward_loss(r, BEVLoss(grid=r.grid), batch, 0.5, "cpu", "off", TrainConfig(radar_aux=1.0))
    rr = r(batch, dts=dts)
check("radar_side" not in rr and "radar_aux" not in b and float(a["total"]) == float(b["total"]),
      "radar_kind raster: no radar_side, radar_aux changes nothing (identical loss)")

print("RADAR SLOT GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)

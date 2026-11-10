"""Radar-only BEV decoder (Mirko, 2026-10-25): stationary environment + moving objects from radar alone, no camera.

Same techniques as the fusion decoder (plan.md sections 4-8), only the camera branch is missing:
- time: every detection of the last sweeps is moved ONCE to the output time with the splat mathematics
  (training/splat.py chain_positions): exact ego motion of its sweep's steps plus its measured radial motion
  (Doppler along the original ray); the ray direction is rotated into the output axes; its age dt is kept
- encoder: training/radar_bev.py (detections into their cells with count, mean properties - range, ray direction,
  Doppler as a circular code, RCS, age - and sub-cell offset; two 3 x 3 convs)
- trunk: training/bev_trunk.py (3-stage ResNet pyramid + FPN back to 0.5 m)
- heads: a stationary-map head (1 logit per cell: occupied by something that does not move - road edges, guard rails,
  parked objects) and the object head of training/object_head.py for MOVING objects (oriented boxes, circles, velocity)
Run the self-test: python training/legacy/radar_decoder.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

_TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # moved to training/legacy/ (2026-10-31): the live blocks
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
from bev_trunk import BEVTrunk  # noqa: E402
from lift_pd import Grid  # noqa: E402
from object_head import GROUPS, ObjectHead  # noqa: E402
from radar_bev import RadarBEV  # noqa: E402
from splat import chain_positions  # noqa: E402


def to_output_time(xy, psi, vr, steps, use_doppler=True):
    """Detections of one sweep (xy (N, 2) and ray direction psi (N,) in that sweep's ego frame, radial velocity vr (N,))
    -> positions and ray directions in the output frame. steps: [(dt, v, omega), ...] from the sweep to the output
    time (see splat.chain_positions); the radial motion vr * u is the only object motion radar measures."""
    if not steps:
        return xy, psi
    u = torch.stack([torch.cos(psi), torch.sin(psi)], -1)
    v_obj = (vr[:, None] * u)[None] if use_doppler else None
    pos = chain_positions(xy[None], steps, v_obj)[0]
    rot = sum(float(w) * float(dt) for dt, _, w in steps)
    return pos, psi - rot


class RadarOnlyDecoder(nn.Module):
    """det (dict of tensors b, x, y, r, psi, vr, rcs, dt - already in the output frame) -> dict: 'static' (B, 1, X, Y)
    logits and the object head's maps (moving objects)."""

    def __init__(self, grid=None, enc_channels=32, width=0.5, groups=GROUPS):
        super().__init__()
        self.grid = grid or Grid()
        self.encoder = RadarBEV(self.grid, enc_channels)
        self.trunk = BEVTrunk(cin=enc_channels, width=width)
        c = self.trunk.img_bev_encoder_neck.out_channels
        self.static_head = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.static_head[2].bias, -2.19)
        self.objects = ObjectHead(c, groups)

    def forward(self, det, B):
        f = self.trunk.features(self.encoder(det, B))
        out = self.objects(f)
        out["static"] = self.static_head(f)
        return out


def _selftest():
    torch.manual_seed(0)
    grid = Grid()
    # a static detection 0.25 s old: ego moved 2.5 m forward -> it is 2.5 m closer; a moving one with vr = 4 m/s on a
    # ray straight ahead: additionally 1 m further out (radial motion only)
    xy = torch.tensor([[30.0, 2.0], [30.0, 0.0]])
    psi = torch.tensor([math.atan2(2.0, 30.0), 0.0])
    vr = torch.tensor([0.0, 4.0])
    steps = [(0.125, 10.0, 0.0), (0.125, 10.0, 0.0)]
    pos, psi2 = to_output_time(xy, psi, vr, steps)
    assert torch.allclose(pos, torch.tensor([[27.5, 2.0], [28.5, 0.0]]), atol=1e-5), pos
    assert torch.allclose(psi2, psi)
    pos_t, psi_t = to_output_time(xy, psi, vr, [(0.25, 0.0, 0.4)])           # pure rotation: directions rotate along
    assert abs((psi - psi_t)[1].item() - 0.1) < 1e-6
    model = RadarOnlyDecoder(grid)
    det = {"b": torch.tensor([0, 0]), "x": pos[:, 0], "y": pos[:, 1], "r": pos.norm(dim=1), "psi": psi2, "vr": vr,
           "rcs": torch.zeros(2), "dt": torch.tensor([0.25, 0.25])}
    out = model(det, 1)
    assert out["static"].shape == (1, 1, grid.nx, grid.ny) and out["hm"].shape == (1, len(GROUPS), grid.nx, grid.ny)
    print("radar_decoder selftest ok: static detection moved by ego motion only, moving one also by its radial "
          "motion, directions rotated with the ego, decoder shapes")


if __name__ == "__main__":
    _selftest()

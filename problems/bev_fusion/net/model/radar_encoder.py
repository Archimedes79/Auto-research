"""Radar encoder that keeps the information (the owner, 2026-10-26; the codec study is in the tag v1).

Replaces nothing: net/model/radar_bev.py stays (the simple block); this is the variant the unit test asked for.

What it stores per detection (a detection is range / direction seen from the sensor, so x, y, r, psi are only TWO numbers once the
ray origin s = p - r u, the sensor position at the measurement time, is known):
  valid | dr, dpsi | Doppler as a circular code | rcs | age | sub-cell offset (ox, oy) | ray origin s (explicit side information)
  position = cell index (implicit in the tensor position, exact) + offset (explicit delta, |o| <= 0.5 cell)
  dr   = r - |c - s|              residual of the distance against what the cell and the ray origin already imply (|dr| <= 0.36 m)
  dpsi = psi - bearing(c - s)     residual of the direction                                                   (c = cell centre)
  r and psi are decoded as |c - s| + dr and bearing(c - s) + dpsi: exact, nothing is lost. K = 3 detections per cell exact (slots),
  the rest of the cell as a remainder mean (rasterize_delta_slots / decode_slots / unpack).
The ray origin is taken from the detection dict ("sx", "sy") or derived as (x - r cos psi, y - r sin psi); an adapter that moves older
sweeps by Doppler should supply the true origin.

Learned part (RadarEncoder): two 3 x 3 convs with LeakyReLU(0.5) (invertible: a ReLU deletes the negative half of every signed
channel), C >= the number of input channels, the first convs START AS A PASS-THROUGH (centre tap = identity + small noise): the net only
has to keep the information, a random start has to find it (25 % kept from the default start, 94 % from this start, the codec study in the tag v1).

Side loss, not the goal: net/loss/radar_loss.slot_aux_loss reconstructs the stored fields from the features (1 x 1 decoder, tolerance-weighted: every channel
is weighted by 1 / the tolerance of its field, so the capacity goes where the sensor is precise). It helps the first phase of the
training - the features start informative and cannot collapse onto what the first task gradients happen to need. The final goal
is the task (segmentation, objects, velocity): radar_loss.aux_weight(step) decays the side loss to a small floor, and `kept_fraction` measures
(no gradient) how much of the detection cloud the features still hold, so one can SEE what the task training does to the information.
`forward(..., with_state=True)` also returns the lossless raster (the explicit stream, bypass for heads that read raw values).

Tests: tests/test_radar_encoder.py
"""
import math

import torch
import torch.nn as nn

from bev_fusion.net.model import radar_bev as rb
from bev_fusion.net.model.lift_pd import Grid

K_SLOTS = 3
DR_SCALE, DPSI_SCALE, S_SCALE = 1.0, 0.05, 4.0           # metres, radians, metres: stored values are O(1)
TOLERANCES = {"pos (m)": 0.25, "dist (m)": 0.5, "dir (deg)": 3.0, "vr (m/s)": 0.5, "rcs (dB)": 3.0, "age (s)": 0.1}
KEYS = ("x", "y", "r", "psi", "vr", "rcs", "dt")


def slot_width(harmonics=rb.HARMONICS):
    """channels per slot: valid, dr, dpsi, Doppler code, rcs, age, ox, oy, sx, sy."""
    return 1 + 2 + 2 * len(harmonics) + 2 + 2 + 2


def n_channels(K=K_SLOTS, harmonics=rb.HARMONICS):
    return 1 + (K + 1) * slot_width(harmonics)


def tolerance_weights(harmonics=rb.HARMONICS, v_max=rb.V_MAX):
    """loss weight of every value channel of a slot (valid excluded, ray origin 0): 1 / tolerance in the stored unit, mean 5."""
    tol = [TOLERANCES["dist (m)"] / DR_SCALE, math.radians(TOLERANCES["dir (deg)"]) / DPSI_SCALE]
    tol += [k * math.pi * TOLERANCES["vr (m/s)"] / v_max for k in harmonics for _ in (0, 1)]
    tol += [TOLERANCES["rcs (dB)"] / rb.RCS_SCALE, TOLERANCES["age (s)"] / rb.AGE_SCALE, 0.5, 0.5]
    w = torch.tensor([1.0 / t for t in tol])
    w = 5.0 * w / w.mean()
    return torch.cat([w, torch.zeros(2)])


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def ray_origin(det):
    """sensor position at the measurement time in the output frame: the supplied (sx, sy) or p - r u."""
    if "sx" in det:
        return det["sx"], det["sy"]
    return det["x"] - det["r"] * torch.cos(det["psi"]), det["y"] - det["r"] * torch.sin(det["psi"])


def _slots(det, grid, K):
    """detections inside the grid, their cell, and the slot rank in the cell (newest first, then nearest to the sensor)."""
    fx, fy = (det["x"] - grid.x_min) / grid.res, (det["y"] - grid.y_min) / grid.res
    ix, iy = fx.floor().long(), fy.floor().long()
    inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    idx = torch.nonzero(inside).squeeze(1)
    ix, iy = ix[idx], iy[idx]
    cell = (det["b"][idx] * grid.nx + ix) * grid.ny + iy
    order = torch.argsort(det["r"][idx], stable=True)
    order = order[torch.argsort(det["dt"][idx][order], stable=True)]
    order = order[torch.argsort(cell[order], stable=True)]
    _, inv, cnt = torch.unique_consecutive(cell[order], return_inverse=True, return_counts=True)
    rank_sorted = torch.arange(len(order), device=cell.device) - (torch.cumsum(cnt, 0) - cnt)[inv]
    rank = torch.empty_like(idx)
    rank[order] = rank_sorted
    return idx, ix, iy, cell, rank, fx[idx] - ix - 0.5, fy[idx] - iy - 0.5


def _values(det, idx, ix, iy, ox, oy, grid, v_max, harmonics):
    """(n, slot_width) stored values of the detections idx."""
    d = {k: det[k][idx] for k in KEYS}
    sx, sy = ray_origin({k: v[idx] for k, v in det.items() if k in KEYS + ("sx", "sy")})
    cx, cy = grid.x_min + (ix + 0.5) * grid.res, grid.y_min + (iy + 0.5) * grid.res
    rho, beta = torch.hypot(cx - sx, cy - sy), torch.atan2(cy - sy, cx - sx)
    return torch.cat([torch.ones_like(cx)[:, None], ((d["r"] - rho) / DR_SCALE)[:, None],
                      (_wrap(d["psi"] - beta) / DPSI_SCALE)[:, None], rb.doppler_code(d["vr"], v_max, harmonics),
                      (d["rcs"] / rb.RCS_SCALE)[:, None], (d["dt"] / rb.AGE_SCALE)[:, None], ox[:, None], oy[:, None],
                      (sx / S_SCALE)[:, None], (sy / S_SCALE)[:, None]], -1)


def rasterize_delta_slots(det, B, grid=None, K=K_SLOTS, v_max=rb.V_MAX, harmonics=rb.HARMONICS):
    """Detections -> (B, 1 + (K + 1) W, NX, NY): channel 0 = number of detections in the cell, then K exact slots and a remainder
    slot (mean of the rest), each [valid, dr, dpsi, Doppler code, rcs, age, ox, oy, sx, sy]. No learned parameters."""
    grid = grid or Grid()
    W = slot_width(harmonics)
    idx, ix, iy, cell, rank, ox, oy = _slots(det, grid, K)
    vals = _values(det, idx, ix, iy, ox, oy, grid, v_max, harmonics)
    dev = vals.device
    out = torch.zeros(B * grid.nx * grid.ny, 1 + (K + 1) * W, device=dev, dtype=vals.dtype)
    out[:, 0].index_add_(0, cell, torch.ones(len(cell), device=dev, dtype=vals.dtype))
    for k in range(K):
        m = rank == k
        out[cell[m], 1 + k * W: 1 + (k + 1) * W] = vals[m]
    m = rank >= K
    rem = torch.zeros(B * grid.nx * grid.ny, W, device=dev, dtype=vals.dtype)
    rem.index_add_(0, cell[m], vals[m])
    out[:, 1 + K * W:] = torch.cat([(rem[:, :1] > 0).to(vals.dtype), rem[:, 1:] / rem[:, :1].clamp(min=1)], 1)
    return out.view(B, grid.nx, grid.ny, -1).permute(0, 3, 1, 2).contiguous()


def unpack(x, grid=None, K=K_SLOTS, v_max=rb.V_MAX, harmonics=rb.HARMONICS, state=None):
    """(B, 1 + (K + 1) W, NX, NY) -> physical fields of every slot, dense: dict of (B, K + 1, NX, NY) tensors: valid (raw channel:
    probability for the raster, logit for a decoder output), x, y, r, psi, vr, rcs, dt. The ray origin is read from `state` (the
    explicit raster) when given - a decoder output need not reproduce it - else from x itself."""
    grid = grid or Grid()
    W = slot_width(harmonics)
    nd = 2 * len(harmonics)
    assert x.shape[1] == n_channels(K, harmonics), "channel count does not match K and the harmonics"
    assert harmonics[0] == 1
    src = x if state is None else state
    cx = (grid.x_min + (torch.arange(grid.nx, device=x.device, dtype=x.dtype) + 0.5) * grid.res)[None, :, None]
    cy = (grid.y_min + (torch.arange(grid.ny, device=x.device, dtype=x.dtype) + 0.5) * grid.res)[None, None, :]
    out = {k: [] for k in ("valid", "x", "y", "r", "psi", "vr", "rcs", "dt")}
    for k in range(K + 1):
        v = x[:, 1 + k * W: 1 + (k + 1) * W]
        s = src[:, 1 + k * W: 1 + (k + 1) * W]
        sx, sy = s[:, W - 2] * S_SCALE, s[:, W - 1] * S_SCALE
        dx, dy = cx - sx, cy - sy
        out["valid"].append(v[:, 0])
        out["x"].append(cx + v[:, W - 4] * grid.res)
        out["y"].append(cy + v[:, W - 3] * grid.res)
        out["r"].append(torch.hypot(dx, dy) + v[:, 1] * DR_SCALE)
        out["psi"].append(torch.atan2(dy, dx) + v[:, 2] * DPSI_SCALE)
        out["vr"].append(torch.atan2(v[:, 3], v[:, 4]) * v_max / math.pi)
        out["rcs"].append(v[:, 3 + nd] * rb.RCS_SCALE)
        out["dt"].append(v[:, 4 + nd] * rb.AGE_SCALE)
    return {k: torch.stack(v, 1) for k, v in out.items()}


def decode_slots(state, grid=None, K=K_SLOTS, v_max=rb.V_MAX, harmonics=rb.HARMONICS):
    """Inverse of rasterize_delta_slots for the K exact slots: the detections they hold (b, x, y, r, psi, vr, rcs, dt) and the
    remainder (cells with more than K detections: b, ix, iy, n = count - K, mean fields). vr is exact for |vr| < v_max."""
    grid = grid or Grid()
    u = unpack(state, grid, K, v_max, harmonics)
    valid = u["valid"] > 0.5
    b = torch.arange(state.shape[0], device=state.device)[:, None, None, None].expand_as(valid)
    det = {"b": b[:, :K][valid[:, :K]]}
    for k in ("x", "y", "r", "psi", "vr", "rcs", "dt"):
        det[k] = u[k][:, :K][valid[:, :K]]
    rv = valid[:, K]
    bi, ix, iy = torch.nonzero(rv, as_tuple=True)
    rem = {"b": bi, "ix": ix, "iy": iy, "n": state[:, 0][rv] - K}
    for k in ("x", "y", "r", "psi", "vr", "rcs", "dt"):
        rem[k] = u[k][:, K][rv]
    return det, rem


def _pass_through_(conv):
    """centre tap = identity on the first min(out, in) channels, small noise elsewhere, zero bias."""
    with torch.no_grad():
        conv.weight.mul_(0.05)
        for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
            conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
        conv.bias.zero_()


class RadarEncoder(nn.Module):
    """Delta-coded slot raster -> two 3 x 3 convs (pass-through start, LeakyReLU) -> (B, C, NX, NY) on the camera BEV grid.

    forward(det, B, with_state=False) -> features [, state]; radar_loss.slot_aux_loss(module, features, state) = the side loss (tolerance-weighted
    reconstruction of the stored fields); kept_fraction(det, B, features) = share of the detections whose fields are all within
    tolerance after decoding the features (measurement only)."""

    def __init__(self, grid=None, C=64, K=K_SLOTS, v_max=rb.V_MAX, harmonics=rb.HARMONICS, identity_init=True):
        super().__init__()
        self.grid, self.K, self.v_max, self.harmonics = grid or Grid(), K, v_max, harmonics
        self.cin = n_channels(K, harmonics)
        assert C >= self.cin, f"C = {C} < {self.cin} input channels: a pass-through needs at least one channel per input channel"
        A = lambda: nn.LeakyReLU(0.5)
        self.net = nn.Sequential(nn.Conv2d(self.cin, C, 3, padding=1), A(), nn.Conv2d(C, C, 3, padding=1), A())
        self.decoder = nn.Sequential(nn.Conv2d(C, C, 1), A(), nn.Conv2d(C, self.cin, 1))      # side loss only, never feeds the BEV
        if identity_init:
            for conv in (self.net[0], self.net[2], self.decoder[0], self.decoder[2]):
                _pass_through_(conv)
        self.register_buffer("w_slot", tolerance_weights(harmonics, v_max), persistent=False)

    def rasterize(self, det, B):
        return rasterize_delta_slots(det, B, self.grid, self.K, self.v_max, self.harmonics)

    def forward(self, det, B, with_state=False):
        state = self.rasterize(det, B)
        f = self.net(state)
        return (f, state) if with_state else f

    @torch.no_grad()
    def kept_fraction(self, det, B, features=None, decoder=None):
        """Share of the detections (inside the grid) whose own slot - or the remainder mean of a crowded cell - decodes to all fields
        within the tolerances, from the decoder applied to `features` (from the raster itself when None: the ceiling), and the
        share violating each field. A measurement, not a loss."""
        state = self.rasterize(det, B)
        x = state if features is None else (decoder or self.decoder)(features)
        u = unpack(x, self.grid, self.K, self.v_max, self.harmonics, state=state)
        idx, ix, iy, cell, rank, _, _ = _slots(det, self.grid, self.K)
        b, slot = det["b"][idx], rank.clamp(max=self.K)
        g = lambda t: t[b, slot, ix, iy]
        valid = g(u["valid"]) > (0.5 if features is None else 0.0)
        d = {k: det[k][idx] for k in KEYS}
        e = {"pos (m)": torch.hypot(g(u["x"]) - d["x"], g(u["y"]) - d["y"]), "dist (m)": (g(u["r"]) - d["r"]).abs(),
             "dir (deg)": torch.rad2deg(_wrap(g(u["psi"]) - d["psi"]).abs()),
             "vr (m/s)": (torch.remainder(g(u["vr"]) - d["vr"] + self.v_max, 2 * self.v_max) - self.v_max).abs(),
             "rcs (dB)": (g(u["rcs"]) - d["rcs"]).abs(), "age (s)": (g(u["dt"]) - d["dt"]).abs()}
        ok = valid.clone()
        per_field = {}
        for k, t in TOLERANCES.items():
            within = valid & (e[k] <= t)
            per_field[k] = 1.0 - within.float().mean().item() if len(within) else 0.0
            ok &= e[k] <= t
        return (ok.float().mean().item() if len(ok) else 1.0), per_field


# ----------------------------------------------------------------------------------------------- self-test

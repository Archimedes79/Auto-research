"""Radar encoder that keeps the information (Mirko, 2026-10-26; plan.md section 3.4, studies/early/radar_codecs.py).

Replaces nothing: training/radar_bev.py stays (the simple block); this is the variant the unit test asked for.

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
has to keep the information, a random start has to find it (25 % kept from the default start, 94 % from this start, studies/early/radar_codecs.py).

Side loss, not the goal: `aux_loss` reconstructs the stored fields from the features (1 x 1 decoder, tolerance-weighted: every channel
is weighted by 1 / the tolerance of its field, so the capacity goes where the sensor is precise). It helps the first phase of the
training - the features start informative and cannot collapse onto what the first task gradients happen to need. The final goal
is the task (segmentation, objects, velocity): `aux_weight(step)` decays the side loss to a small floor, and `kept_fraction` measures
(no gradient) how much of the detection cloud the features still hold, so one can SEE what the task training does to the information.
`forward(..., with_state=True)` also returns the lossless raster (the explicit stream, bypass for heads that read raw values).

Run the self-test: python training/radar_encoder.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import radar_bev as rb
    from lift_pd import Grid
except ImportError:  # imported as a package module
    from . import radar_bev as rb
    from .lift_pd import Grid

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


def aux_weight(step, start=1.0, floor=0.05, half_life=3000):
    """Weight of the side loss at a training step: starts at `start`, halves every `half_life` steps down to `floor` (0 switches the
    side loss off at the end). The task losses are the goal; this only keeps the first phase informative."""
    return floor + (start - floor) * 0.5 ** (step / half_life)


def _pass_through_(conv):
    """centre tap = identity on the first min(out, in) channels, small noise elsewhere, zero bias."""
    with torch.no_grad():
        conv.weight.mul_(0.05)
        for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
            conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
        conv.bias.zero_()


class RadarEncoder(nn.Module):
    """Delta-coded slot raster -> two 3 x 3 convs (pass-through start, LeakyReLU) -> (B, C, NX, NY) on the camera BEV grid.

    forward(det, B, with_state=False) -> features [, state]; aux_loss(features, state) = the side loss (tolerance-weighted
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

    def new_decoder(self, C=None, identity_init=True):
        """A fresh 1 x 1 decoder for probing frozen features (same start as the module's own decoder: a probe that has to find the
        identity from a random start under-measures): the honest way to measure what a task-trained encoder kept."""
        C = C or self.net[2].out_channels
        dec = nn.Sequential(nn.Conv2d(C, C, 1), nn.LeakyReLU(0.5), nn.Conv2d(C, self.cin, 1)).to(self.w_slot.device)
        if identity_init:
            _pass_through_(dec[0])
            _pass_through_(dec[2])
        return dec

    def aux_loss(self, features, state, decoder=None):
        """Side loss: count (L1 on log1p), valid (BCE) and the tolerance-weighted L1 of the stored fields at valid slots."""
        out = (decoder or self.decoder)(features)
        W = slot_width(self.harmonics)
        loss = F.l1_loss(out[:, 0], torch.log1p(state[:, 0]))
        w = self.w_slot.view(1, -1, 1, 1)
        for k in range(self.K + 1):
            vi = 1 + k * W
            valid = (state[:, vi] > 0.5).float()
            loss = loss + F.binary_cross_entropy_with_logits(out[:, vi], valid)
            m = valid[:, None]
            loss = loss + ((out[:, vi + 1: vi + W] - state[:, vi + 1: vi + W]).abs() * w * m).sum() / (m.sum() * (W - 1)).clamp(min=1)
        return loss

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
def _synthetic(n, B, grid, g, cells=None):
    """detections from sensors at random poses: polar (r, phi) -> vehicle frame, so the geometry is consistent by construction."""
    sx = torch.randint(0, 3, (n,), generator=g).float() * 1.3 - 0.5
    sy = (torch.randint(0, 3, (n,), generator=g).float() - 1) * 0.8
    yaw = (torch.rand(n, generator=g) - 0.5) * 3.0
    r = 5 + 55 * torch.rand(n, generator=g)
    phi = (torch.rand(n, generator=g) - 0.5) * 2.0
    x, y, psi = rb.to_vehicle(r, phi, sx, sy, yaw)
    if cells is not None:
        x = grid.x_min + (cells[:, 0] + 0.5 + (torch.rand(n, generator=g) - 0.5) * 0.9) * grid.res
        y = grid.y_min + (cells[:, 1] + 0.5 + (torch.rand(n, generator=g) - 0.5) * 0.9) * grid.res
        r = torch.hypot(x - sx, y - sy)
        psi = torch.atan2(y - sy, x - sx)
    return {"b": torch.randint(0, B, (n,), generator=g), "x": x, "y": y, "r": r, "psi": psi,
            "vr": (torch.rand(n, generator=g) - 0.5) * 55, "rcs": 30 * (torch.rand(n, generator=g) - 0.5), "dt": 0.5 * torch.rand(n, generator=g)}


def _as_rows(d):
    rows = torch.stack([d["b"].float(), d["x"], d["y"], d["r"], torch.remainder(d["psi"], 2 * math.pi), d["vr"], d["rcs"], d["dt"]], 1)
    return rows[torch.argsort(rows[:, 1] * 1000 + rows[:, 2])]


def _selftest():
    grid = Grid()
    g = torch.Generator().manual_seed(3)
    # 1 exact round trip of all eight fields: distinct cells, two frames, sensors at different poses
    n = 150
    ix = torch.arange(n)
    cells = torch.stack([(ix * 5) % 120, (ix * 7) % 104], 1)
    d = _synthetic(n, 2, grid, g, cells=cells)
    d["b"] = ix % 2
    s = rasterize_delta_slots(d, 2, grid)
    assert s.shape == (2, n_channels(), 120, 104) and n_channels() == 61
    dec, rem = decode_slots(s, grid)
    a, b = _as_rows(d), _as_rows(dec)
    assert len(a) == len(b) == n and rem["n"].numel() == 0
    err = (a - b).abs()
    err[:, 4] = torch.remainder(a[:, 4] - b[:, 4] + math.pi, 2 * math.pi).sub(math.pi).abs()      # psi modulo 2 pi
    assert err.max() < 2e-4, f"round trip {err.max(0).values}"
    # 2 the residuals are small whatever the sensor pose: |dr| <= half a cell diagonal, |dpsi| bounded by the cell seen from r
    valid = s[:, 1] > 0.5
    assert (s[:, 2][valid].abs() * DR_SCALE).max() < 0.36, "dr bounded by the half cell diagonal"
    # 3 several detections in one cell (up to K exact), overflow = K exact + remainder mean and count
    for n_in in (1, 3, 5):
        d = _synthetic(n_in, 1, grid, g, cells=torch.tensor([[40, 50]] * n_in))
        d["b"] = torch.zeros(n_in, dtype=torch.long)
        dec, rem = decode_slots(rasterize_delta_slots(d, 1, grid), grid)
        assert len(dec["x"]) == min(n_in, 3) and (rem["n"].tolist() == [float(n_in - 3)] if n_in > 3 else rem["n"].numel() == 0)
        if n_in > 3:
            order = torch.argsort(torch.argsort(d["dt"]))
            assert {round(v, 4) for v in dec["r"].tolist()} == {round(v, 4) for v in d["r"][order < 3].tolist()}, "the K newest are exact"
            assert abs(rem["r"][0].item() - d["r"][order >= 3].mean().item()) < 0.15, "remainder = mean of the rest (mean origin: curvature term)"
    # 4 Doppler near the wrap; supplied ray origin is used
    d = _synthetic(2, 1, grid, g, cells=torch.tensor([[30, 30], [31, 30]]))
    d["b"] = torch.zeros(2, dtype=torch.long)
    d["vr"] = torch.tensor([29.9, -29.9])
    dec, _ = decode_slots(rasterize_delta_slots(d, 1, grid), grid)
    assert sorted(round(v, 3) for v in dec["vr"].tolist()) == [-29.9, 29.9]
    d2 = {**d, "sx": d["x"] - d["r"] * torch.cos(d["psi"]) + 1.0, "sy": d["y"] - d["r"] * torch.sin(d["psi"]) - 0.5}     # another origin
    dec2, _ = decode_slots(rasterize_delta_slots(d2, 1, grid), grid)
    assert (_as_rows(dec2) - _as_rows(d)).abs()[:, [1, 2, 3]].max() < 2e-4, "a supplied ray origin is carried, r is still exact"
    # 5 the module: shapes, the side loss reaches every parameter, the ceiling decodes from the raster itself, the kept-fraction monitor
    m = RadarEncoder(C=64)
    d = _synthetic(600, 2, grid, g)
    f, st = m(d, 2, with_state=True)
    assert f.shape == (2, 64, 120, 104) and torch.equal(st, m.rasterize(d, 2))
    loss = m.aux_loss(f, st)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    ceil, _ = m.kept_fraction(d, 2)
    start, _ = m.kept_fraction(d, 2, features=f.detach())
    assert 0.9 < ceil <= 1.0, f"the raster decodes {ceil}"
    # 6 the pass-through start: the side loss of the untrained module is lower than that of a default-start net
    plain = RadarEncoder(C=64, identity_init=False)
    loss_plain = plain.aux_loss(plain(d, 2).detach(), st).item()
    assert loss.item() < loss_plain, f"identity start {loss.item():.3f} vs default start {loss_plain:.3f}"
    # 7 aux weight schedule: from `start` to the floor, never below it
    assert abs(aux_weight(0) - 1.0) < 1e-9 and abs(aux_weight(3000) - 0.525) < 1e-9 and aux_weight(10 ** 6) >= 0.05
    print(f"radar_encoder selftest ok: exact round trip of eight fields (K = 3, {n_channels()} channels), residuals bounded, overflow + "
          f"remainder, Doppler wrap, module + side loss gradients; side loss at the start: identity "
          f"{loss.item():.3f} vs default {loss_plain:.3f}; raster decodes {ceil:.3f}")


if __name__ == "__main__":
    _selftest()

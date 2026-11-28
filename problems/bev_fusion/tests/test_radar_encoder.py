"""Tests of net/model/radar_encoder.py (its former self-test).

    python tests/test_radar_encoder.py      (from the problem folder, with its module path)
"""
import math

import torch

from net.model import radar_bev as rb
from net.model.radar_encoder import DR_SCALE, Grid, RadarEncoder, decode_slots, n_channels, rasterize_delta_slots


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
    from net.loss.radar_loss import aux_weight, slot_aux_loss
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
    loss = slot_aux_loss(m, f, st)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    ceil, _ = m.kept_fraction(d, 2)
    start, _ = m.kept_fraction(d, 2, features=f.detach())
    assert 0.9 < ceil <= 1.0, f"the raster decodes {ceil}"
    # 6 the pass-through start: the side loss of the untrained module is lower than that of a default-start net
    plain = RadarEncoder(C=64, identity_init=False)
    loss_plain = slot_aux_loss(plain, plain(d, 2).detach(), st).item()
    assert loss.item() < loss_plain, f"identity start {loss.item():.3f} vs default start {loss_plain:.3f}"
    # 7 aux weight schedule: from `start` to the floor, never below it
    assert abs(aux_weight(0) - 1.0) < 1e-9 and abs(aux_weight(3000) - 0.525) < 1e-9 and aux_weight(10 ** 6) >= 0.05
    print(f"radar_encoder selftest ok: exact round trip of eight fields (K = 3, {n_channels()} channels), residuals bounded, overflow + "
          f"remainder, Doppler wrap, module + side loss gradients; side loss at the start: identity "
          f"{loss.item():.3f} vs default {loss_plain:.3f}; raster decodes {ceil:.3f}")


if __name__ == "__main__":
    _selftest()

"""Radar block (simple, Mirko 2026-10-25): every detection goes into the BEV cell where it resides, together with its
properties, followed by two 3x3 convolutions. No pillars, no learned footprint, no polar image (those stay later variants,
see bev_fusion/radar_encoding.md).

Per detection (all supplied by an adapter, in vehicle coordinates, so any rig works):
  x, y   position in the vehicle frame at the OUTPUT time (metres)       r    distance from the sensor (metres)
  psi    direction from the sensor to the detection in the vehicle frame at the time of the measurement (radians): the
         ORIGINAL ray direction - after an ego-motion shift of an older sweep it is not recomputed from x, y
  vr     radial (Doppler) velocity along that ray (m/s); the adapter decides what it is (ego-compensated for nuScenes)
  rcs    radar cross section (dBsm)                                      dt   age at the output time (s)
  b      index of the frame (batch) the detection belongs to
Properties per detection (11 values with the default harmonics):
  r / 60 | cos psi, sin psi | Doppler as a CIRCULAR value: sin / cos of k pi vr / v_max for k in harmonics (a radar measures
  Doppler only modulo its unambiguous interval, so +v_max and -v_max are neighbours) | rcs / 20 | dt / 0.5
Raster channels per cell of the grid (x forward, y left, 0.5 m, same grid and index order as the camera BEV (x, y)):
  count | mean of the properties of the detections in the cell | mean sub-cell offset (dx, dy in cell units): nothing is lost
  by binning.  Cells without a detection are zero.

Run the self-test: python training/radar_bev.py
"""
import math

import torch
import torch.nn as nn

try:
    from lift_pd import Grid
except ImportError:  # imported as a package module
    from .lift_pd import Grid

V_MAX = 30.0                 # unambiguous Doppler interval half-width in m/s (sensor property; nuScenes: 30 above the observed range)
HARMONICS = (1, 2, 4)        # all integers: the code stays periodic with the interval
RANGE_SCALE, RCS_SCALE, AGE_SCALE = 60.0, 20.0, 0.5


def to_vehicle(r, phi, sx, sy, yaw):
    """Sensor polar (range, azimuth in the sensor frame) -> vehicle-frame position and ray direction psi."""
    psi = phi + yaw
    return sx + r * torch.cos(psi), sy + r * torch.sin(psi), psi


def doppler_code(vr, v_max=V_MAX, harmonics=HARMONICS):
    """(n,) m/s -> (n, 2 * len(harmonics)): sin / cos of k pi vr / v_max. vr and vr + 2 v_max have the same code."""
    a = math.pi * vr / v_max
    return torch.cat([torch.stack([torch.sin(k * a), torch.cos(k * a)], -1) for k in harmonics], -1)


def properties(det, v_max=V_MAX, harmonics=HARMONICS):
    return torch.cat([(det["r"] / RANGE_SCALE)[:, None], torch.cos(det["psi"])[:, None], torch.sin(det["psi"])[:, None],
                      doppler_code(det["vr"], v_max, harmonics), (det["rcs"] / RCS_SCALE)[:, None],
                      (det["dt"] / AGE_SCALE)[:, None]], -1)


def n_properties(harmonics=HARMONICS):
    return 5 + 2 * len(harmonics)


def rasterize(det, B, grid=None, v_max=V_MAX, harmonics=HARMONICS):
    """Detections -> (B, 1 + P + 2, NX, NY): count, mean properties, mean sub-cell offset. No learned parameters."""
    grid = grid or Grid()
    NX, NY, res = grid.nx, grid.ny, grid.res
    P = n_properties(harmonics)
    fx, fy = (det["x"] - grid.x_min) / res, (det["y"] - grid.y_min) / res
    ix, iy = fx.floor().long(), fy.floor().long()
    inside = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    vals = torch.cat([torch.ones_like(fx)[:, None], properties(det, v_max, harmonics),
                      (fx - ix - 0.5)[:, None], (fy - iy - 0.5)[:, None]], -1)[inside]
    flat = (det["b"][inside] * NX + ix[inside]) * NY + iy[inside]
    acc = torch.zeros(B * NX * NY, 1 + P + 2, device=vals.device, dtype=vals.dtype)
    acc.index_add_(0, flat, vals)
    cnt = acc[:, :1]
    acc = torch.cat([cnt, acc[:, 1:] / cnt.clamp(min=1)], 1)
    return acc.view(B, NX, NY, 1 + P + 2).permute(0, 3, 1, 2).contiguous()


class RadarBEV(nn.Module):
    """Raster + two 3x3 convolutions -> (B, out_channels, NX, NY), the grid of the camera BEV."""

    def __init__(self, grid=None, out_channels=32, v_max=V_MAX, harmonics=HARMONICS):
        super().__init__()
        self.grid, self.v_max, self.harmonics = grid or Grid(), v_max, harmonics
        cin = 1 + n_properties(harmonics) + 2
        self.net = nn.Sequential(nn.Conv2d(cin, out_channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def forward(self, det, B):
        return self.net(rasterize(det, B, self.grid, self.v_max, self.harmonics))


# ----------------------------------------------------------------------------------------------- self-test
def _det(b, x, y, r, psi, vr, rcs=0.0, dt=0.0):
    n = len(x)
    t = lambda v: torch.as_tensor(v, dtype=torch.float32).reshape(-1).expand(n).clone()
    return {"b": torch.tensor(b), "x": t(x), "y": t(y), "r": t(r), "psi": t(psi), "vr": t(vr), "rcs": t(rcs), "dt": t(dt)}


def _selftest():
    grid = Grid()
    NX, NY = grid.nx, grid.ny
    assert (NX, NY) == (120, 104)
    # 1 Doppler code is periodic with the unambiguous interval and +-v_max are neighbours
    v = torch.linspace(-V_MAX, V_MAX, 41)
    assert torch.allclose(doppler_code(v + 2 * V_MAX), doppler_code(v), atol=1e-5), "periodic"
    assert torch.allclose(doppler_code(torch.tensor([V_MAX])), doppler_code(torch.tensor([-V_MAX])), atol=1e-5)
    c = doppler_code(torch.tensor([0.0]))
    assert torch.allclose(c, torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]), atol=1e-6), "zero velocity"
    # 2 a radar mounted 2 m to the left, yawed 90 deg, sees a target straight ahead of itself at 10 m
    x, y, psi = to_vehicle(torch.tensor([10.0]), torch.tensor([0.0]), 0.0, 2.0, math.pi / 2)
    assert abs(x.item()) < 1e-5 and abs(y.item() - 12.0) < 1e-5 and abs(psi.item() - math.pi / 2) < 1e-6
    # 3 a detection lands in the cell it resides in, with its properties and sub-cell offset
    d = _det([0], [20.30], [1.10], [20.4], [0.1], [3.0], [10.0], [0.25])
    out = rasterize(d, 1, grid)
    ix, iy = int(20.30 / 0.5), int((1.10 + 26.0) / 0.5)
    assert out.shape == (1, 14, NX, NY) and out[0, 0, ix, iy] == 1 and out[0, 0].sum() == 1
    p = properties(d)[0]
    assert torch.allclose(out[0, 1:12, ix, iy], p, atol=1e-6)
    assert abs(out[0, 12, ix, iy].item() - (20.30 / 0.5 - ix - 0.5)) < 1e-5 and abs(out[0, 13, ix, iy].item() - ((1.10 + 26.0) / 0.5 - iy - 0.5)) < 1e-5
    assert out[0, :, ix + 1, iy].abs().sum() == 0, "other cells empty"
    # 4 outside the grid is dropped; two detections in one cell give the mean; frames do not mix
    d = _det([0, 0, 1, 0], [20.1, 20.3, 20.2, 70.0], [1.1, 1.2, 1.1, 0.0], [20.0] * 4, [0.0] * 4, [2.0, 4.0, 9.0, 0.0])
    out = rasterize(d, 2, grid)
    assert out[0, 0].sum() == 2 and out[1, 0].sum() == 1 and out[0, 0, 40, 54] == 2
    a = math.pi * torch.tensor([2.0, 4.0]) / V_MAX
    assert abs(out[0, 4, 40, 54].item() - torch.sin(a).mean().item()) < 1e-6, "mean of the circular code"
    # 5 rig independence of occupancy: two sensors see the same point; count and position channels are equal, the
    #   sensor-relative channels (distance, direction) differ as they must
    xa, ya, pa = to_vehicle(torch.tensor([15.0]), torch.tensor([0.0]), 0.0, 0.0, 0.0)             # sensor at the origin
    r2 = torch.hypot(xa - 1.0, ya - 3.0)
    ph2 = torch.atan2(ya - 3.0, xa - 1.0)                                                           # sensor at (1, 3), yaw 0
    xb, yb, pb = to_vehicle(r2, ph2, 1.0, 3.0, 0.0)
    assert abs((xa - xb).item()) < 1e-5 and abs((ya - yb).item()) < 1e-5
    da = {"b": torch.tensor([0]), "x": xa, "y": ya, "r": torch.tensor([15.0]), "psi": pa, "vr": torch.tensor([1.0]), "rcs": torch.tensor([0.0]), "dt": torch.tensor([0.0])}
    db = {**da, "r": r2, "psi": pb}
    oa, ob = rasterize(da, 1, grid), rasterize(db, 1, grid)
    assert torch.equal(oa[:, 0], ob[:, 0]) and torch.allclose(oa[:, -2:], ob[:, -2:]), "same cell, same offset"
    assert (oa[:, 1:3] - ob[:, 1:3]).abs().max() > 0.01, "distance / direction differ"
    # 6 the module: shapes, gradient reaches the convolutions, no learned parameter in the raster
    m = RadarBEV(out_channels=32)
    y_ = m(d, 2)
    assert y_.shape == (2, 32, NX, NY)
    y_.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print("radar_bev selftest ok: periodic Doppler code, sensor pose, cell and properties, mean of a cell, frames, rig independence, module")
    if torch.cuda.is_available():
        import time
        dev = "cuda"
        n = 750
        g = torch.Generator(device=dev).manual_seed(0)
        det = {"b": torch.zeros(n, dtype=torch.long, device=dev), "x": 60 * torch.rand(n, device=dev, generator=g),
               "y": -26 + 52 * torch.rand(n, device=dev, generator=g), "r": 60 * torch.rand(n, device=dev, generator=g),
               "psi": torch.rand(n, device=dev, generator=g), "vr": 20 * torch.randn(n, device=dev, generator=g),
               "rcs": torch.randn(n, device=dev, generator=g), "dt": 0.5 * torch.rand(n, device=dev, generator=g)}
        mm = RadarBEV().to(dev)
        for _ in range(10):
            mm(det, 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            mm(det, 1)
        torch.cuda.synchronize()
        print(f"radar block (raster + 2 convs, 750 detections, 120 x 104) on {torch.cuda.get_device_name(0)}: "
              f"{(time.perf_counter() - t0) * 10:.2f} ms per frame (GPU may be shared - not a timing number)")


if __name__ == "__main__":
    _selftest()

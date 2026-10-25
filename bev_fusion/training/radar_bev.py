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


def _slot_values(det, ix, iy, inside, grid, v_max, harmonics):
    fx, fy = (det["x"] - grid.x_min) / grid.res, (det["y"] - grid.y_min) / grid.res
    return torch.cat([torch.ones_like(fx)[:, None], properties(det, v_max, harmonics),
                      (fx - ix - 0.5)[:, None], (fy - iy - 0.5)[:, None]], -1)


def rasterize_slots(det, B, grid=None, K=3, v_max=V_MAX, harmonics=HARMONICS):
    """Information-preserving raster: the first K detections of a cell are stored EXACTLY, one slot each (newest first, then
    nearest to the sensor), the rest of the cell is summarised by the mean of its detections (remainder slot).

    Returns (B, 1 + (K + 1) * W, NX, NY) with W = 1 + P + 2: channel 0 = number of detections in the cell (overflow is visible);
    then K slots and the remainder slot, each [valid, properties (P), sub-cell offset (2)]. decode_slots() inverts it.
    """
    grid = grid or Grid()
    NX, NY = grid.nx, grid.ny
    P = n_properties(harmonics)
    W = 1 + P + 2
    fx, fy = (det["x"] - grid.x_min) / grid.res, (det["y"] - grid.y_min) / grid.res
    ix, iy = fx.floor().long(), fy.floor().long()
    inside = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    d = {k: v[inside] for k, v in det.items()}
    ix, iy = ix[inside], iy[inside]
    vals = _slot_values(d, ix, iy, inside, grid, v_max, harmonics)
    cell = (d["b"] * NX + ix) * NY + iy
    order = torch.argsort(d["r"], stable=True)                       # three stable sorts: cell, then age, then distance
    order = order[torch.argsort(d["dt"][order], stable=True)]
    order = order[torch.argsort(cell[order], stable=True)]
    vals, cell = vals[order], cell[order]
    _, inv, cnt = torch.unique_consecutive(cell, return_inverse=True, return_counts=True)
    rank = torch.arange(len(cell), device=cell.device) - (torch.cumsum(cnt, 0) - cnt)[inv]
    out = torch.zeros(B * NX * NY, 1 + (K + 1) * W, device=vals.device, dtype=vals.dtype)
    out[:, 0].index_add_(0, cell, torch.ones(len(cell), device=vals.device, dtype=vals.dtype))
    for k in range(K):
        m = rank == k
        out[cell[m], 1 + k * W: 1 + (k + 1) * W] = vals[m]
    m = rank >= K
    rem = torch.zeros(B * NX * NY, W, device=vals.device, dtype=vals.dtype)
    rem.index_add_(0, cell[m], vals[m])
    rem = torch.cat([(rem[:, :1] > 0).to(vals.dtype), rem[:, 1:] / rem[:, :1].clamp(min=1)], 1)
    out[:, 1 + K * W:] = rem
    return out.view(B, NX, NY, -1).permute(0, 3, 1, 2).contiguous()


def decode_slots(r, grid=None, K=3, v_max=V_MAX, harmonics=HARMONICS):
    """Inverse of rasterize_slots for the K exact slots: returns the detections (b, x, y, r, psi, vr, rcs, dt) they hold, plus the
    remainder (cells with more than K detections: their count n - K and mean properties are in the returned 'remainder' dict).
    vr is recovered modulo the unambiguous interval (exact for |vr| < v_max)."""
    grid = grid or Grid()
    P = n_properties(harmonics)
    W = 1 + P + 2
    B, C, NX, NY = r.shape
    assert C == 1 + (K + 1) * W, "channel count does not match K and the harmonics"
    x = r.permute(0, 2, 3, 1)                                         # (B, NX, NY, C)
    bi, xi, yi = torch.meshgrid(torch.arange(B, device=r.device), torch.arange(NX, device=r.device), torch.arange(NY, device=r.device), indexing="ij")
    outs = {k: [] for k in ["b", "x", "y", "r", "psi", "vr", "rcs", "dt"]}
    def unpack(s, valid):
        v = s[valid]                                                  # (n, W)
        p = v[:, 1:1 + P]
        ph = torch.atan2(p[:, 3], p[:, 4])                            # k = 1 pair of the Doppler code (harmonics[0] must be 1)
        return v, p, ph
    assert harmonics[0] == 1
    for k in range(K):
        s = x[..., 1 + k * W: 1 + (k + 1) * W]
        valid = s[..., 0] > 0.5
        v, p, ph = unpack(s, valid)
        outs["b"].append(bi[valid]); outs["r"].append(p[:, 0] * RANGE_SCALE); outs["psi"].append(torch.atan2(p[:, 2], p[:, 1]))
        outs["vr"].append(ph * v_max / math.pi); outs["rcs"].append(p[:, 3 + 2 * len(harmonics)] * RCS_SCALE)
        outs["dt"].append(p[:, 4 + 2 * len(harmonics)] * AGE_SCALE)
        outs["x"].append(grid.x_min + (xi[valid].to(v.dtype) + 0.5 + v[:, 1 + P]) * grid.res)
        outs["y"].append(grid.y_min + (yi[valid].to(v.dtype) + 0.5 + v[:, 2 + P]) * grid.res)
    det = {k: torch.cat(v) for k, v in outs.items()}
    s = x[..., 1 + K * W:]
    valid = s[..., 0] > 0.5
    rem = {"b": bi[valid], "ix": xi[valid], "iy": yi[valid], "n": x[..., 0][valid] - K, "props": s[valid][:, 1:1 + P]}
    return det, rem


class RadarBEVSlots(nn.Module):
    """Information-preserving radar block: slot raster (exact for K detections per cell) + two 3x3 convolutions.
    forward(det, B) -> features (B, out_channels, NX, NY); with_state=True also returns the raster (the lossless state)."""

    def __init__(self, grid=None, out_channels=32, K=3, v_max=V_MAX, harmonics=HARMONICS):
        super().__init__()
        self.grid, self.K, self.v_max, self.harmonics = grid or Grid(), K, v_max, harmonics
        cin = 1 + (K + 1) * (1 + n_properties(harmonics) + 2)
        self.net = nn.Sequential(nn.Conv2d(cin, out_channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def forward(self, det, B, with_state=False):
        r = rasterize_slots(det, B, self.grid, self.K, self.v_max, self.harmonics)
        f = self.net(r)
        return (f, r) if with_state else f


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


def _selftest_slots():
    grid = Grid()
    g = torch.Generator().manual_seed(1)

    def rand_det(n, b=None, cells=None):
        x = 60 * torch.rand(n, generator=g)
        y = -26 + 52 * torch.rand(n, generator=g)
        if cells is not None:                                      # force detections into given cells (x, y of the cell centre + jitter)
            x = (torch.as_tensor(cells)[:, 0] + 0.5 + (torch.rand(n, generator=g) - 0.5) * 0.9) * 0.5
            y = -26 + (torch.as_tensor(cells)[:, 1] + 0.5 + (torch.rand(n, generator=g) - 0.5) * 0.9) * 0.5
        return {"b": torch.zeros(n, dtype=torch.long) if b is None else b, "x": x, "y": y,
                "r": 3 + 60 * torch.rand(n, generator=g), "psi": (torch.rand(n, generator=g) - 0.5) * 6.28,
                "vr": (torch.rand(n, generator=g) - 0.5) * 50, "rcs": 30 * (torch.rand(n, generator=g) - 0.5),
                "dt": 0.5 * torch.rand(n, generator=g)}

    def as_set(d, nd=3):
        rows = torch.stack([d["b"].float(), d["x"], d["y"], d["r"], d["psi"], d["vr"], d["rcs"], d["dt"]], 1)
        return rows[torch.argsort(rows[:, 1] * 1000 + rows[:, 2])]

    # 1 distinct cells: the raster is exactly invertible (all eight fields, two frames)
    ix = torch.arange(100)
    cells = torch.stack([ix % 120, (ix * 7) % 104], 1)
    d = rand_det(100, b=(ix % 2), cells=cells)
    r = rasterize_slots(d, 2, grid, K=3)
    assert r.shape == (2, 1 + 4 * 14, 120, 104)
    dec, rem = decode_slots(r, grid, K=3)
    a, b = as_set(d), as_set(dec)
    assert len(a) == len(b) == 100 and (a - b).abs().max() < 2e-4, f"round trip {(a - b).abs().max()}"
    assert rem["n"].numel() == 0
    # 2 several detections in one cell (up to K): all recovered individually; K + 2: K exact + a remainder with count and mean
    for n_in in (2, 3, 5):
        d = rand_det(n_in, cells=torch.tensor([[40, 50]] * n_in))
        dec, rem = decode_slots(rasterize_slots(d, 1, grid, K=3), grid, K=3)
        assert len(dec["x"]) == min(n_in, 3)
        if n_in > 3:
            assert rem["n"].tolist() == [float(n_in - 3)] and rem["ix"].tolist() == [40]
            kept = torch.argsort(torch.argsort(d["dt"]))             # newest (smallest dt) first
            first = {round(v, 4) for v in d["r"][kept < 3].tolist()}
            assert {round(v, 4) for v in dec["r"].tolist()} == first, "the K newest are kept exactly"
            mean_r = d["r"][kept >= 3].mean() / RANGE_SCALE
            assert abs(rem["props"][0, 0].item() - mean_r.item()) < 1e-5, "remainder is the mean of the rest"
        else:
            assert (as_set(d) - as_set(dec)).abs().max() < 2e-4
    # 3 the count channel shows the overflow; the module returns features and the lossless state
    m = RadarBEVSlots(out_channels=32, K=3)
    f, st = m(d, 1, with_state=True)
    assert f.shape == (1, 32, 120, 104) and st[0, 0, 40, 50] == 5
    f.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print("radar_bev slots selftest ok: exact round trip of eight fields (K = 3), several detections per cell, overflow count and remainder mean, module")


if __name__ == "__main__":
    _selftest()
    _selftest_slots()

"""Tests of net/model/radar_bev.py (its former self-test).

    python tests/test_radar_bev.py      (from the problem folder, with its module path)
"""
import math

import torch

from net.model.radar_bev import Grid, RadarBEV, V_MAX, doppler_code, properties, rasterize, to_vehicle


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

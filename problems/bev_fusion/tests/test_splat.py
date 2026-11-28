"""Tests of net/model/splat.py (its former self-test).

    python tests/test_splat.py      (from the problem folder, with its module path)
"""
import math

import torch

from net.model.lift_pd import Grid
from net.model.splat import VelocityHead, chain_positions, ego_speed_yawrate, grid_points, splat, splat_bev, splat_positions, splat_window


def _exact_ctrv(x, y, dt, v, w):
    th = w * dt
    dx, dy = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
    qx, qy = x - dx, y - dy
    return math.cos(th) * qx + math.sin(th) * qy, -math.sin(th) * qx + math.cos(th) * qy


def _centroid(mass, grid):
    xs, ys = grid.xy_centres(mass.device, mass.dtype)
    m = mass[0, 0]
    return (m.sum(1) @ xs / m.sum()).item(), (m.sum(0) @ ys / m.sum()).item()


def _point_mass(bev, grid, v_hat):
    """mass of the non-zero cells of bev only (helper of test 8)."""
    B, C, X, Y = bev.shape
    p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0)
    nz = bev.reshape(B, C, -1)[0].abs().sum(0) > 0
    v_obj = None if v_hat is None else v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2)[:, nz]
    pos = splat_positions(p[:, nz], 0.5, 0.0, 0.0, v_obj)
    _, m = splat(torch.ones(1, 1, int(nz.sum()), dtype=bev.dtype), pos, grid)
    return m


def _selftest():
    torch.manual_seed(0)
    dt64 = torch.float64
    grid = Grid()

    # 1 pure translation by exactly 1 m forward (omega = 0, v_hat = 0): the map moves back by exactly 2 cells
    bev = torch.randn(1, 3, grid.nx, grid.ny, dtype=dt64)
    out, mass, _ = splat_bev(bev, grid, dt=0.1, v=10.0, omega=0.0)
    assert torch.allclose(out[:, :, :-2], bev[:, :, 2:], atol=1e-9) and torch.allclose(mass[0, 0, :-2], torch.ones_like(mass[0, 0, :-2]))
    assert (mass[0, 0, -2:] == 0).all(), "cells nothing landed in have mass 0"

    # 2 a single point lands where the formula says (the bilinear mass centroid is exact) and matches the exact
    #   constant-turn-rate motion within the first-order translation error
    for (x, y, dt, v, w) in [(30.0, 5.0, 1 / 12, 30.0, 0.2), (40.0, -10.0, 0.5, 8.0, 0.5), (20.0, 0.0, -0.3, 15.0, -0.3)]:
        p = torch.tensor([[[x, y]]], dtype=dt64)
        pos = splat_positions(p, dt, v, w)
        _, m = splat(torch.ones(1, 1, 1, dtype=dt64), pos, grid)
        cx, cy = _centroid(m, grid)
        assert abs(cx - pos[0, 0, 0]) < 1e-9 and abs(cy - pos[0, 0, 1]) < 1e-9, "centroid"
        ex, ey = _exact_ctrv(x, y, dt, v, w)
        assert math.hypot(pos[0, 0, 0] - ex, pos[0, 0, 1] - ey) < 0.2 * v * w * w * abs(dt) ** 3 + 1e-9, "vs exact"

    # 3 object velocity: a point with v_obj = (3, -2) m/s and no ego motion lands at p + v dt; forward then back = identity
    p = torch.tensor([[[25.0, 4.0]]], dtype=dt64)
    vo = torch.tensor([[[3.0, -2.0]]], dtype=dt64)
    pos = splat_positions(p, 0.5, 0.0, 0.0, vo)
    assert torch.allclose(pos, torch.tensor([[[26.5, 3.0]]], dtype=dt64))
    back = splat_positions(splat_positions(p, 0.4, 12.0, 0.3), -0.4, 12.0, 0.3)
    assert torch.allclose(back, p, atol=0.02), "forward then backward"

    # 4 mass conservation: every source inside the grid contributes exactly 1
    pts = torch.stack([torch.rand(500, dtype=dt64) * 50 + 5, torch.rand(500, dtype=dt64) * 40 - 20], -1)[None]
    _, m = splat(torch.ones(1, 1, 500, dtype=dt64), pts, grid)
    assert abs(m.sum().item() - 500) < 1e-9

    # 5 collision: two sources on the same spot, the more important one wins
    two = torch.tensor([[[30.0, 0.0], [30.0, 0.0]]], dtype=dt64)
    f = torch.tensor([[[1.0, -1.0]]], dtype=dt64)
    o, _ = splat(f, two, grid, importance=torch.tensor([[8.0, 0.0]], dtype=dt64))
    assert o.max() > 0.99, "importance"

    # 6 gradients reach features, positions (hence v_obj, ego values) and importance
    feat = torch.randn(1, 2, 7, dtype=dt64, requires_grad=True)
    pos = (torch.rand(1, 7, 2, dtype=dt64) * torch.tensor([10.0, 6.0], dtype=dt64) + torch.tensor([10.0, -3.0], dtype=dt64)).requires_grad_(True)
    imp = torch.randn(1, 7, dtype=dt64, requires_grad=True)
    small = Grid(x_min=8, x_max=22, y_min=-5, y_max=5, res=1.0)
    assert torch.autograd.gradcheck(lambda a, b, c: splat(a, b, small, c)[0], (feat, pos, imp), eps=1e-6, atol=1e-5)

    # 7 the velocity head starts at zero (pure ego compensation) and learns a velocity from the feature
    head = VelocityHead(cin=4).double()
    fe = torch.randn(64, 4, 1, 1, dtype=dt64)
    assert head(fe).abs().max() == 0, "zero start"
    target = torch.cat([2.0 * fe[:, :1] - fe[:, 1:2], 3.0 * fe[:, 2:3]], 1) * 5
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(600):
        opt.zero_grad()
        loss = ((head(fe) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * (target ** 2).mean().item(), f"velocity head did not learn ({loss.item():.3f})"

    # 8 a moving blob: with the learned head's velocity the blob lands at x + v dt, without it (v_hat = 0) it lags
    bev = torch.zeros(1, 1, grid.nx, grid.ny, dtype=dt64)
    bev[0, 0, 60, 52] = 1.0                                   # cell centre (30.25, 0.25)
    head2 = VelocityHead(cin=1).double()
    head2.net[2].bias.data = torch.tensor([8.0, 0.0], dtype=dt64)   # constant 8 m/s forward (v_max tanh keeps ~8)
    out_v, _, vh = splat_bev(bev, grid, 0.5, 0.0, 0.0, head=head2)        # the full map path runs
    assert out_v.shape == bev.shape
    cx_v, _ = _centroid(_point_mass(bev, grid, vh), grid)                  # where the blob's own cell lands
    v_used = 40.0 * math.tanh(8.0 / 40.0)
    assert abs(cx_v - (30.25 + v_used * 0.5)) < 1e-6, "moving blob with velocity"
    assert abs(_centroid(_point_mass(bev, grid, None), grid)[0] - 30.25) < 1e-9, "without velocity it stays (lags)"
    # 9 speed and yaw rate from poses: an exact constant-turn-rate drive is recovered (v within the first-order
    #   translation model, omega exactly), and the splat with them moves static points like the poses do
    v_true, w_true, dt = 12.0, 0.4, 1 / 12
    th = w_true * dt
    T0 = torch.eye(4, dtype=dt64)
    T1 = torch.eye(4, dtype=dt64)
    T1[:2, :2] = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=dt64)
    T1[0, 3], T1[1, 3] = v_true / w_true * math.sin(th), v_true / w_true * (1 - math.cos(th))
    v_est, w_est = ego_speed_yawrate(T0, T1, dt)
    assert abs(w_est.item() - w_true) < 1e-12 and abs(v_est.item() - v_true) < v_true * th * th / 6 + 1e-12
    pw = torch.tensor([[[35.0, -6.0]]], dtype=dt64)              # static point in frame 0
    via_pose = (torch.linalg.inv(T1) @ torch.tensor([35.0, -6.0, 0.0, 1.0], dtype=dt64))[:2]
    via_splat = splat_positions(pw, dt, v_est, w_est)[0, 0]
    assert (via_splat - via_pose).norm() < 1e-3, "splat with v, omega from poses = pose transform"
    # 11 coordinate chaining over varying steps (speed and yaw rate change every frame) against the composition of the
    #    exact per-step motions; an object with a velocity constant in the world is followed exactly up to the
    #    first-order translation error of each step
    steps = [(1 / 12, 8.0 + i, 0.5 - 0.15 * i) for i in range(6)]
    ex, ey, eh = 0.0, 0.0, 0.0                                   # exact ego pose in the source frame (= world)
    for dt_i, v_i, w_i in steps:
        th = w_i * dt_i
        lx, ly = (v_i * dt_i, 0.0) if abs(w_i) < 1e-12 else (v_i / w_i * math.sin(th), v_i / w_i * (1 - math.cos(th)))
        ex, ey = ex + math.cos(eh) * lx - math.sin(eh) * ly, ey + math.sin(eh) * lx + math.cos(eh) * ly
        eh += th
    T = sum(s[0] for s in steps)
    p0 = torch.tensor([[[40.0, 5.0], [20.0, -8.0]]], dtype=dt64)
    vo = torch.tensor([[[6.0, -1.0], [0.0, 0.0]]], dtype=dt64)
    got = chain_positions(p0, steps, vo)
    for j in range(2):
        wx, wy = p0[0, j, 0].item() + vo[0, j, 0].item() * T, p0[0, j, 1].item() + vo[0, j, 1].item() * T
        qx, qy = wx - ex, wy - ey
        tx, ty = math.cos(eh) * qx + math.sin(eh) * qy, -math.sin(eh) * qx + math.cos(eh) * qy
        assert math.hypot(got[0, j, 0] - tx, got[0, j, 1] - ty) < 0.01, f"chained coordinates {j}"

    # 12 window: two frames of the same static scene, one 2 frames old, land on the same map as the current frame
    st = [(0.1, 10.0, 0.0), (0.1, 10.0, 0.0)]                    # 2 m forward in total = 4 cells
    old = torch.zeros(1, 1, grid.nx, grid.ny, dtype=dt64)
    old[0, 0, 60, 30] = 1.0                                       # a static point at x = 30.25 m in the old frame
    cur = torch.zeros_like(old)
    cur[0, 0, 56, 30] = 1.0                                       # the same point now: 2 m closer
    out_w, mass_w = splat_window([(old, st), (cur, [])], grid)
    assert torch.allclose(out_w[0, 0, 0] * (mass_w[0, 0, 0] > 0), out_w[0, 1, 0] * (mass_w[0, 1, 0] > 0)), "window"

    print("splat selftest ok: exact 2-cell shift, landing point = formula (vs exact CTRV within v w^2 dt^3), "
          "object velocity, forward-back, mass conservation, collision by importance, gradcheck, velocity head learns, "
          "moving blob")


if __name__ == "__main__":
    _selftest()

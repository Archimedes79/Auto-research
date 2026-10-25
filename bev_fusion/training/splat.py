"""Splat in time: move BEV cells or radar detections to another time by ego motion + learned object motion.

One function serves the camera history (past camera BEVs -> current camera time), the radar history (past sweeps ->
current radar time) and the hop of the radar BEV into the camera time (bev_fusion/plan.md section 7, Mirko's
per-branch design):

    v_hat = f_theta(F(x, y))                         velocity head: 1x1-conv MLP on the feature, 2 outputs (m/s)
    q     = p + v_hat * dt - d(dt)                   object motion, minus the ego displacement
    d     = (v * dt, 1/2 * v * omega * dt^2)         ego displacement, first order in omega (no division by omega)
    p'    = R(-omega * dt) q                         ego rotation, exact cos / sin (one pair per frame)

with p in the source ego frame (x forward, y left), v the ego speed, omega the ego yaw rate, dt = t_target - t_source
(either sign). Measured (temp/setup/check_ctrv_taylor.py): with an exact rotation the remaining translation error is of
order v omega^2 dt^3 (millimetres per camera frame); a first-order rotation would cost up to distance * (omega dt)^2 / 2.

The splat (forward warp) writes each source feature to the 4 cells around p' with bilinear weights k and an optional
importance z (softmax splatting: the more important source wins a collision), normalised:
    out(q) = sum_i e^z_i k_i(q) f_i / (sum_i e^z_i k_i(q) + eps),   mass(q) = sum_i k_i(q)
mass is 0 where nothing landed (holes are visible) and serves as the confidence channel of the fusion.
Basis: softmax splatting (Niklaus & Liu, CVPR 2020; from memory), CRT-Fusion's push (literature/code/crt_fusion.md:
collisions overwrite there), FIERY's ego warp (literature/code/fiery_ego_warp.md). Plain PyTorch.
Run the self-test: python training/splat.py
"""
import math

import torch
import torch.nn as nn

try:
    from .lift_pd import Grid
except ImportError:                                   # run as a script
    from lift_pd import Grid


def grid_points(grid, device=None, dtype=torch.float32):
    """Cell centres (X*Y, 2) in the order of a flattened (X, Y) map (x-major)."""
    xs, ys = grid.xy_centres(device, dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)


def splat_positions(p, dt, v, omega, v_obj=None):
    """p (B, N, 2) source positions; dt, v, omega: (B,) or (B, N); v_obj (B, N, 2) or None -> p' (B, N, 2)."""
    def col(t):
        t = torch.as_tensor(t, dtype=p.dtype, device=p.device)
        return t.view(-1, 1) if t.dim() <= 1 else t
    dt, v, omega = col(dt), col(v), col(omega)
    qx = p[..., 0] - v * dt
    qy = p[..., 1] - 0.5 * v * omega * dt * dt
    if v_obj is not None:
        qx = qx + v_obj[..., 0] * dt
        qy = qy + v_obj[..., 1] * dt
    th = omega * dt
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * qx + s * qy, -s * qx + c * qy], -1)


def ego_speed_yawrate(T_src, T_tgt, dt):
    """Ego speed v and yaw rate omega between two ego poses (4x4, ego -> world; batch (B, 4, 4) or single) over dt,
    consistent with splat_positions: omega = heading change / dt, v = forward displacement in the source frame / dt.
    A vehicle reports v and omega directly (wheel speed, yaw-rate sensor); nuScenes gives poses, hence this helper."""
    rel = torch.linalg.inv(T_src) @ T_tgt                       # target ego pose in the source ego frame
    dth = torch.atan2(rel[..., 1, 0], rel[..., 0, 0])
    dt = torch.as_tensor(dt, dtype=rel.dtype, device=rel.device)
    return rel[..., 0, 3] / dt, dth / dt


class VelocityHead(nn.Module):
    """f_theta: per-cell MLP as 1x1 convs, feature (B, C, ...) -> object velocity (B, 2, ...) in m/s, source axes.

    The last layer starts at zero (v_hat = 0: pure ego compensation at the start); v_hat = v_max tanh(raw / v_max).
    """

    def __init__(self, cin, hidden=64, v_max=40.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.v_max = v_max

    def forward(self, feat):
        squeeze = feat.dim() == 3                          # points (B, C, N)
        x = feat.unsqueeze(-1) if squeeze else feat
        raw = self.net(x)
        out = self.v_max * torch.tanh(raw / self.v_max)
        return out.squeeze(-1) if squeeze else out


def splat(feat, pos, grid, importance=None, eps=1e-6):
    """feat (B, C, N), pos (B, N, 2) target positions in metres, importance (B, N) or None -> (out (B, C, X, Y),
    mass (B, 1, X, Y)). Bilinear kernel on the cell centres; sources whose 4 corners all fall outside are dropped."""
    B, C, N = feat.shape
    X, Y = grid.nx, grid.ny
    fx = (pos[..., 0] - grid.x_min) / grid.res - 0.5           # continuous cell index
    fy = (pos[..., 1] - grid.y_min) / grid.res - 0.5
    x0, y0 = torch.floor(fx), torch.floor(fy)
    ax, ay = fx - x0, fy - y0
    w_imp = torch.ones_like(fx) if importance is None else torch.exp(importance - importance.max())
    num = feat.new_zeros(B, C, X * Y)
    den = feat.new_zeros(B, X * Y)
    mass = feat.new_zeros(B, X * Y)
    boff = (torch.arange(B, device=feat.device) * X * Y).view(B, 1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix, iy = x0 + dx, y0 + dy
        k = (ax if dx else 1 - ax) * (ay if dy else 1 - ay)
        ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
        k = k * ok
        idx = (ix.clamp(0, X - 1) * Y + iy.clamp(0, Y - 1)).long()
        flat = (idx + boff).reshape(-1)
        wk = (k * w_imp).reshape(-1)
        num_flat = num.permute(1, 0, 2).reshape(C, B * X * Y)
        num_flat = num_flat.index_add(1, flat, feat.permute(1, 0, 2).reshape(C, -1) * wk)
        num = num_flat.reshape(C, B, X * Y).permute(1, 0, 2)
        den = den.reshape(-1).index_add(0, flat, wk).reshape(B, X * Y)
        mass = mass.reshape(-1).index_add(0, flat, k.reshape(-1)).reshape(B, X * Y)
    out = num / (den.unsqueeze(1) + eps)
    return out.view(B, C, X, Y), mass.view(B, 1, X, Y)


def splat_bev(bev, grid, dt, v, omega, head=None, importance=None):
    """Move a BEV map (B, C, X, Y) of the source time to the target time: positions of the cell centres by
    splat_positions with v_hat = head(bev) (or 0), then splat. Returns (out, mass, v_hat or None)."""
    B, C, X, Y = bev.shape
    p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0).expand(B, -1, -1)
    v_hat = head(bev) if head is not None else None
    v_obj = v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2) if v_hat is not None else None
    pos = splat_positions(p, dt, v, omega, v_obj)
    imp = importance.reshape(B, X * Y) if importance is not None else None
    out, mass = splat(bev.reshape(B, C, X * Y), pos, grid, imp)
    return out, mass, v_hat


# ----------------------------------------------------------------------------------------------- self-test
def _exact_ctrv(x, y, dt, v, w):
    th = w * dt
    dx, dy = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
    qx, qy = x - dx, y - dy
    return math.cos(th) * qx + math.sin(th) * qy, -math.sin(th) * qx + math.cos(th) * qy


def _centroid(mass, grid):
    xs, ys = grid.xy_centres(mass.device, mass.dtype)
    m = mass[0, 0]
    return (m.sum(1) @ xs / m.sum()).item(), (m.sum(0) @ ys / m.sum()).item()


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
    print("splat selftest ok: exact 2-cell shift, landing point = formula (vs exact CTRV within v w^2 dt^3), "
          "object velocity, forward-back, mass conservation, collision by importance, gradcheck, velocity head learns, "
          "moving blob")


def _point_mass(bev, grid, v_hat):
    """mass of the non-zero cells of bev only (helper of test 8)."""
    B, C, X, Y = bev.shape
    p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0)
    nz = bev.reshape(B, C, -1)[0].abs().sum(0) > 0
    v_obj = None if v_hat is None else v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2)[:, nz]
    pos = splat_positions(p[:, nz], 0.5, 0.0, 0.0, v_obj)
    _, m = splat(torch.ones(1, 1, int(nz.sum()), dtype=bev.dtype), pos, grid)
    return m


if __name__ == "__main__":
    _selftest()

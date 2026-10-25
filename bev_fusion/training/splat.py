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


def splat_positions(p, dt, v, omega, v_obj=None, delta=None):
    """p (B, N, 2) source positions; dt, v, omega: (B,) or (B, N); v_obj (B, N, 2) or None; delta (B, N, 2) a learned
    correction in the target frame (CorrectionHead) or None -> p' (B, N, 2)."""
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
    out = torch.stack([c * qx + s * qy, -s * qx + c * qy], -1)
    return out if delta is None else out + delta


def chain_positions(p, steps, v_obj=None, delta=None):
    """Chain the COORDINATES over several frames (Mirko, 2026-10-25; temp/checks showed that chaining resampled MAPS
    smears and lags): p (B, N, 2) positions in the source frame, steps = [(dt_i, v_i, omega_i), ...] from the source
    frame to the target, oldest first (each value a scalar or (B,)); v_obj (B, N, 2) object velocity in the source
    axes, constant in the world, so its vector is rotated along by R(-omega_i dt_i) at every step; delta (B, N, 2) an
    optional correction added once at the end. The positions are then splatted once (splat_window)."""
    pos, vel = p, v_obj
    for dt, v, omega in steps:
        pos = splat_positions(pos, dt, v, omega, vel)
        if vel is not None:
            th = (torch.as_tensor(omega, dtype=p.dtype, device=p.device)
                  * torch.as_tensor(dt, dtype=p.dtype, device=p.device)).reshape(-1, 1)
            c, s = torch.cos(th), torch.sin(th)
            vel = torch.stack([c * vel[..., 0] + s * vel[..., 1], -s * vel[..., 0] + c * vel[..., 1]], -1)
    return pos if delta is None else pos + delta


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


class CorrectionHead(nn.Module):
    """g_phi: a small learned correction of the target position (Mirko, 2026-10-25), on top of the geometric model:
    delta = d_max tanh(raw / d_max), raw = MLP([feature; v; omega; dt; v dt; omega dt; x / x_scale; y / y_scale]) as
    1x1 convs, 2 outputs in metres (target frame). Bounded to d_max (default 1 m = 2 cells: the range in which the
    bilinear splat gives a usable position gradient) and zero at the start (pure geometric model). It can absorb what
    the model leaves out: lag and blur of chained splats, acceleration, odometry / time-stamp / model errors."""

    N_EGO = 7

    def __init__(self, cin, hidden=32, d_max=1.0, x_scale=60.0, y_scale=26.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin + self.N_EGO, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)
        self.d_max, self.x_scale, self.y_scale = d_max, x_scale, y_scale

    def forward(self, feat, pos, dt, v, omega):
        """feat (B, C, N) and pos (B, N, 2) source positions; dt, v, omega (B,) or scalars -> delta (B, N, 2)."""
        B, C, N = feat.shape

        def col(t):
            return torch.as_tensor(t, dtype=feat.dtype, device=feat.device).reshape(-1, 1).expand(B, N)
        dt, v, omega = col(dt), col(v), col(omega)
        ego = torch.stack([v, omega, dt, v * dt, omega * dt, pos[..., 0] / self.x_scale, pos[..., 1] / self.y_scale], 1)
        raw = self.net(torch.cat([feat, ego], 1).unsqueeze(-1))[..., 0]          # (B, 2, N)
        return (self.d_max * torch.tanh(raw / self.d_max)).permute(0, 2, 1)


def splat(feat, pos, grid, importance=None, eps=1e-6, mass_gate=False):
    """feat (B, C, N), pos (B, N, 2) target positions in metres, importance (B, N) or None -> (out (B, C, X, Y),
    mass (B, 1, X, Y)). Bilinear kernel on the cell centres; sources whose 4 corners all fall outside are dropped.
    mass_gate: out *= min(mass, 1), so cells that only received a weak bilinear tail are not scaled up to full
    strength by the normalisation (helped in every variant of temp/checks/check_splat_coordinates.py)."""
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
    if mass_gate:
        out = out * mass.clamp(max=1.0).unsqueeze(1)
    return out.view(B, C, X, Y), mass.view(B, 1, X, Y)


def splat_bev(bev, grid, dt, v, omega, head=None, importance=None, corr=None):
    """Move a BEV map (B, C, X, Y) of the source time to the target time: positions of the cell centres by
    splat_positions with v_hat = head(bev) (or 0) and the correction corr(...) (or none), then splat.
    Returns (out, mass, v_hat or None)."""
    B, C, X, Y = bev.shape
    p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0).expand(B, -1, -1)
    v_hat = head(bev) if head is not None else None
    v_obj = v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2) if v_hat is not None else None
    delta = corr(bev.reshape(B, C, X * Y), p, dt, v, omega) if corr is not None else None
    pos = splat_positions(p, dt, v, omega, v_obj, delta)
    imp = importance.reshape(B, X * Y) if importance is not None else None
    out, mass = splat(bev.reshape(B, C, X * Y), pos, grid, imp)
    return out, mass, v_hat


def splat_window(frames, grid, head=None, corr=None, importance=None, mass_gate=True):
    """Bring the last K original frames into the current time, each with ONE splat (no resampling chain).

    frames: list of (bev (B, C, X, Y), steps) - a past frame with its own per-frame ego steps to the current time
    (see chain_positions; [] for the current frame itself). Per frame: v_hat = head(bev) from its own features, the cell
    centres chained by chain_positions, the correction corr(...) evaluated with the total dt and the dt-weighted mean
    v, omega of its steps, one splat. importance: callable bev -> (B, X, Y) or None.
    Returns out (B, K, C, X, Y) and mass (B, K, 1, X, Y) for the temporal fusion."""
    outs, masses = [], []
    for bev, steps in frames:
        B, C, X, Y = bev.shape
        p = grid_points(grid, bev.device, bev.dtype).unsqueeze(0).expand(B, -1, -1)
        v_obj = head(bev).permute(0, 2, 3, 1).reshape(B, X * Y, 2) if head is not None else None
        delta = None
        if corr is not None and steps:
            T = sum(float(s[0]) for s in steps)
            v_m = sum(float(s[0]) * float(s[1]) for s in steps) / T
            w_m = sum(float(s[0]) * float(s[2]) for s in steps) / T
            delta = corr(bev.reshape(B, C, X * Y), p, T, v_m, w_m)
        pos = chain_positions(p, steps, v_obj, delta)
        imp = importance(bev).reshape(B, X * Y) if importance is not None else None
        o, m = splat(bev.reshape(B, C, X * Y), pos, grid, imp, mass_gate=mass_gate)
        outs.append(o)
        masses.append(m)
    return torch.stack(outs, 1), torch.stack(masses, 1)


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

    # 10 correction head: zero at the start (no change), bounded by d_max, and it gets a gradient through the splat
    corr = CorrectionHead(cin=3, d_max=1.0).double()
    bev = torch.randn(1, 3, grid.nx, grid.ny, dtype=dt64)
    a, _, _ = splat_bev(bev, grid, 0.3, 10.0, 0.2)
    b, _, _ = splat_bev(bev, grid, 0.3, 10.0, 0.2, corr=corr)
    assert torch.allclose(a, b), "correction starts at zero"
    nn.init.normal_(corr.net[4].weight, std=50.0)
    pts = grid_points(grid, dtype=dt64)[None]
    d = corr(bev.reshape(1, 3, -1), pts, 0.3, 10.0, 0.2)
    assert d.abs().max() <= 1.0 + 1e-12 and d.abs().max() > 0.5, "bounded by d_max"
    nn.init.normal_(corr.net[4].weight, std=1e-2)
    out, _, _ = splat_bev(bev, grid, 0.3, 10.0, 0.2, corr=corr)
    (out[:, 0] * torch.randn_like(out[:, 0])).sum().backward()
    assert corr.net[4].weight.grad is not None and corr.net[4].weight.grad.abs().sum() > 0, "gradient reaches g_phi"
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

"""Splat in time: move BEV cells or radar detections to another time by ego motion + learned object motion.

One function serves the camera history (past camera BEVs -> current camera time), the radar history (past sweeps ->
current radar time) and the hop of the radar BEV into the camera time (the owner's
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
Tests: tests/test_splat.py
"""

import torch
import torch.nn as nn


def grid_points(grid, device=None, dtype=torch.float32):
    """Cell centres (X*Y, 2) in the order of a flattened (X, Y) map (x-major)."""
    xs, ys = grid.xy_centres(device, dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)


def _geo(x):
    """positions and velocities are geometry: float32 at least (bf16 resolves only 0.25 m at 32 - 64 m, fp16 0.03 m); float32 / float64 pass unchanged, bit for
    bit"""
    return x.float() if x is not None and x.dtype in (torch.float16, torch.bfloat16) else x


def _geo_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def splat_positions(p, dt, v, omega, v_obj=None):
    """p (B, N, 2) source positions; dt, v, omega: (B,) or (B, N); v_obj (B, N, 2) or None -> p' (B, N, 2). Half-precision inputs are
    computed in float32."""
    p, v_obj = _geo(p), _geo(v_obj)

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


def chain_positions(p, steps, v_obj=None):
    """Chain the COORDINATES over several frames (the owner, 2026-10-25; a check showed that chaining resampled MAPS
    smears and lags): p (B, N, 2) positions in the source frame, steps = [(dt_i, v_i, omega_i), ...] from the source
    frame to the target, oldest first (each value a scalar or (B,)); v_obj (B, N, 2) object velocity in the source
    axes, constant in the world, so its vector is rotated along by R(-omega_i dt_i) at every step. The positions are then
    splatted once (splat_window)."""
    p, v_obj = _geo(p), _geo(v_obj)
    pos, vel = p, v_obj
    for dt, v, omega in steps:
        pos = splat_positions(pos, dt, v, omega, vel)
        if vel is not None:
            th = (torch.as_tensor(omega, dtype=p.dtype, device=p.device)
                  * torch.as_tensor(dt, dtype=p.dtype, device=p.device)).reshape(-1, 1)
            c, s = torch.cos(th), torch.sin(th)
            vel = torch.stack([c * vel[..., 0] + s * vel[..., 1], -s * vel[..., 0] + c * vel[..., 1]], -1)
    return pos


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


def splat(feat, pos, grid, importance=None, eps=1e-6, mass_gate=False):
    """feat (B, C, N), pos (B, N, 2) target positions in metres, importance (B, N) or None -> (out (B, C, X, Y),
    mass (B, 1, X, Y)). Bilinear kernel on the cell centres; sources whose 4 corners all fall outside are dropped.
    mass_gate: out *= min(mass, 1), so cells that only received a weak bilinear tail are not scaled up to full
    strength by the normalisation (it helped in every variant that was checked)."""
    B, C, N = feat.shape
    X, Y = grid.nx, grid.ny
    if pos.dtype in (torch.float16, torch.bfloat16):           # geometry at least float32 under half-precision autocast (bf16
        pos = pos.float()                                      # integers are exact only up to 256: cell indices overflowed the grid)
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
        k = (k * ok).to(feat.dtype)                             # weights in the feature dtype (geometry stays float32)
        idx = ix.clamp(0, X - 1).long() * Y + iy.clamp(0, Y - 1).long()          # integer math: exact for any grid
        flat = (idx + boff).reshape(-1)
        wk = (k * w_imp).reshape(-1).to(feat.dtype)
        num_flat = num.permute(1, 0, 2).reshape(C, B * X * Y)
        num_flat = num_flat.index_add(1, flat, feat.permute(1, 0, 2).reshape(C, -1) * wk)
        num = num_flat.reshape(C, B, X * Y).permute(1, 0, 2)
        den = den.reshape(-1).index_add(0, flat, wk).reshape(B, X * Y)
        mass = mass.reshape(-1).index_add(0, flat, k.reshape(-1)).reshape(B, X * Y)
    out = num / (den.unsqueeze(1) + eps)
    if mass_gate:
        out = out * mass.clamp(max=1.0).unsqueeze(1)
    return out.view(B, C, X, Y), mass.view(B, 1, X, Y)


def splat_bev(bev, grid, dt, v, omega, head=None):
    """Move a BEV map (B, C, X, Y) of the source time to the target time: positions of the cell centres by
    splat_positions with v_hat = head(bev) (or 0), then splat.
    Returns (out, mass, v_hat or None)."""
    B, C, X, Y = bev.shape
    p = grid_points(grid, bev.device, _geo_dtype(bev.dtype)).unsqueeze(0).expand(B, -1, -1)     # cell centres in float32 even when the BEV is bf16 / fp16
    v_hat = head(bev) if head is not None else None
    v_obj = v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2) if v_hat is not None else None
    pos = splat_positions(p, dt, v, omega, v_obj)
    out, mass = splat(bev.reshape(B, C, X * Y), pos, grid)
    return out, mass, v_hat


def splat_window(frames, grid, head=None, mass_gate=True):
    """Bring the last K original frames into the current time, each with ONE splat (no resampling chain).

    frames: list of (bev (B, C, X, Y), steps) - a past frame with its own per-frame ego steps to the current time
    (see chain_positions; [] for the current frame itself). Per frame: v_hat = head(bev) from its own features, the cell
    centres chained by chain_positions, one splat.
    Returns out (B, K, C, X, Y) and mass (B, K, 1, X, Y) for the temporal fusion."""
    outs, masses = [], []
    for bev, steps in frames:
        B, C, X, Y = bev.shape
        p = grid_points(grid, bev.device, _geo_dtype(bev.dtype)).unsqueeze(0).expand(B, -1, -1)       # float32 cell centres even when the BEV is bf16 / fp16
        v_obj = head(bev).permute(0, 2, 3, 1).reshape(B, X * Y, 2) if head is not None else None
        pos = chain_positions(p, steps, v_obj)
        o, m = splat(bev.reshape(B, C, X * Y), pos, grid, mass_gate=mass_gate)
        outs.append(o)
        masses.append(m)
    return torch.stack(outs, 1), torch.stack(masses, 1)


# ----------------------------------------------------------------------------------------------- self-test

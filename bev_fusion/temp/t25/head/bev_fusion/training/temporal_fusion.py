"""Per-branch temporal fusion (bev_fusion/plan.md section 7.1, Mirko 2026-10-25): the state update of one branch (camera
or radar) at its current time from (1) the current planes, (2) the last K original frames brought into the current time
by `splat.splat_window` (each splatted ONCE, with its own per-frame ego steps, moving content by a learned velocity),
with their mass and age, and (3) a static memory in a WORLD-ALIGNED grid.

Per frame (`TemporalFusion.forward`):
  1. pose      the ego pose in the memory frame (x, y, theta) is advanced by the step (dt, v, omega) of the previous
               frame (same constant-turn-rate geometry as splat_positions) and re-centred by WHOLE cells (lossless)
  2. velocity  v_hat = head(cur) (1x1-conv MLP, zero start); static weight s = exp(-(|v_hat| / v_static)^2): moving
               content propagates through the window but stays out of the memory
  3. window    splat_window(past, ...) -> sources (B, K, C, X, Y) with mass; aggregated per cell with weights
               mass * exp(-age / tau) * relevance (a learned 1x1 conv, sigmoid; tau learned) -> agg, the summed mass and the
               mean age (channels, so the fusion knows how fresh and how complete the sources are)
  4. memory    READ before the write: one bilinear grid_sample of the world memory into the current ego frame
               (sampled as sum(w M) / sum(w), so edges are not dimmed) + its confidence
  5. fusion    conv3x3-BN-ReLU over [cur, agg, mass_sum, age_mean, memory, confidence] -> out (B, c_out, X, Y)
  6. write     one splat of [cur, s, obs_mass] into the memory grid at the new pose; running mean with confidence W:
               W' = min(W exp(-dt / mem_tau) + w, w_max), M' = (M W exp(-dt / mem_tau) + F w) / (W exp(...) + w),
               w = min(mass, 1) * s * obs_mass
Why world-aligned: the memory itself is never resampled (only the new observation is splatted in once and the memory is
read once per frame), so nothing blurs or lags over time - the failure of chained map splats (plan 7.1).
`obs_mass` (B, 1, X, Y) marks where the sensor actually observed (camera field of view, radar coverage); cells
outside write nothing. The state is a dict of tensors; `detach_state` cuts the graph (truncated BPTT).

ONE code path for camera and radar (Mirko): the two branches differ only in the data they pass, not in the module.
Conventions as training/splat.py: x forward, y left, maps (B, C, X, Y).
Run the self-test: python training/temporal_fusion.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import launch_saving as LS
    from .lift_pd import Grid
    from .splat import VelocityHead, grid_points, splat, splat_window
except ImportError:                                   # run as a script
    import launch_saving as LS
    from lift_pd import Grid
    from splat import VelocityHead, grid_points, splat, splat_window


def _col(t, like):
    return torch.as_tensor(t, dtype=like.dtype, device=like.device).reshape(-1)


def advance_pose(pose, dt, v, omega):
    """pose (B, 3) = (x, y, theta) of the ego in the memory frame; the ego moves by d = (v dt, v omega dt^2 / 2) in its own
    frame and turns by omega dt (the geometry of splat.splat_positions)."""
    dt, v, om = _col(dt, pose), _col(v, pose), _col(omega, pose)
    dx, dy = v * dt, 0.5 * v * om * dt * dt
    th = pose[:, 2]
    return torch.stack([pose[:, 0] + torch.cos(th) * dx - torch.sin(th) * dy,
                        pose[:, 1] + torch.sin(th) * dx + torch.cos(th) * dy, th + om * dt], 1)


def as_chain(step, like=None):
    """`step` as a list of (dt, v, omega) triples: None -> [], one triple -> [triple], a list of triples (the per-sweep chain, as
    batch["steps"][k]) as it is. With `like` (a tensor with the batch first) every value becomes a (B,) tensor of its dtype / device."""
    if step is None or not len(step):
        return []
    chain = list(step) if isinstance(step[0], (tuple, list)) else [step]
    if like is None:
        return chain
    return [tuple(torch.as_tensor(t, dtype=like.dtype, device=like.device).reshape(-1).expand(like.shape[0]) for t in s) for s in chain]


def advance_chain(pose, step):
    """the pose after a step or a chain of steps (advance_pose one after the other)"""
    for s in as_chain(step):
        pose = advance_pose(pose, *s)
    return pose


def ego_to_memory(pose, pts):
    """Ego-frame points (N, 2) -> memory-frame positions (B, N, 2)."""
    th = pose[:, 2:3]
    c, s = torch.cos(th), torch.sin(th)
    qx, qy = pts[:, 0].view(1, -1), pts[:, 1].view(1, -1)
    return torch.stack([pose[:, 0:1] + c * qx - s * qy, pose[:, 1:2] + s * qx + c * qy], -1)


def recentre(M, W, pose, res):
    """Shift the memory by whole cells so the ego stays within half a cell of the memory centre (lossless: content moves
    by integer cells, the cells that fall off are dropped, the new ones are empty)."""
    shift = torch.round(pose[:, :2] / res)
    pose = torch.cat([pose[:, :2] - shift * res, pose[:, 2:]], 1)
    Ms, Ws = [], []
    host = shift.cpu().tolist() if LS.on("sync") else None             # D53: one device-to-host read instead of two per sample; int() per element as before (a NaN pose raises as before)
    for b in range(M.shape[0]):
        sx, sy = (int(host[b][0]), int(host[b][1])) if host is not None else (int(shift[b, 0]), int(shift[b, 1]))
        if sx == 0 and sy == 0:
            Ms.append(M[b]), Ws.append(W[b])
            continue
        m, w = M[b], W[b]
        H, Wd = m.shape[-2:]
        if abs(sx) >= H or abs(sy) >= Wd:
            Ms.append(torch.zeros_like(m)), Ws.append(torch.zeros_like(w))
            continue
        m, w = torch.roll(m, (-sx, -sy), (1, 2)), torch.roll(w, (-sx, -sy), (1, 2))
        keep = torch.ones(H, Wd, dtype=m.dtype, device=m.device)
        if sx > 0:
            keep[H - sx:] = 0
        elif sx < 0:
            keep[:-sx] = 0
        if sy > 0:
            keep[:, Wd - sy:] = 0
        elif sy < 0:
            keep[:, :-sy] = 0
        Ms.append(m * keep), Ws.append(w * keep)
    return torch.stack(Ms), torch.stack(Ws), pose


def read_memory(M, W, pose, ego_pts, X, Y, mem_grid):
    """World memory -> the current ego frame (B, C, X, Y) + confidence (B, 1, X, Y); sampled as sum(w M) / sum(w)."""
    B = M.shape[0]
    pos = ego_to_memory(pose, ego_pts)
    nx = 2 * (pos[..., 0] - mem_grid.x_min) / (mem_grid.nx * mem_grid.res) - 1          # along H (dim 2)
    ny = 2 * (pos[..., 1] - mem_grid.y_min) / (mem_grid.ny * mem_grid.res) - 1          # along W (dim 3)
    g = torch.stack([ny, nx], -1).view(B, X, Y, 2)
    out = F.grid_sample(torch.cat([M * W, W], 1), g, mode="bilinear", padding_mode="zeros", align_corners=False)
    den = out[:, -1:]
    return out[:, :-1] / (den + 1e-6), den


class TemporalFusion(nn.Module):
    # w_max 2 (was 8, 2026-10-26): the fuse conv sees conf / w_max; sequence training uses clips of 3 keyframes (at most 2
    # writes before the last read), so with 8 the conf input never exceeded about 0.25 in training but reaches 1.0 on a
    # long drive (out of distribution); with 2 it saturates inside a training clip. Window-only runs never read the memory.
    def __init__(self, c_in, c_out, grid, mem_cells=(256, 256), v_static=1.0, mem_tau=10.0, w_max=2.0, tau_src=0.3,
                 head_hidden=32):
        super().__init__()
        self.grid, self.c_in, self.c_out = grid, c_in, c_out
        self.mem_grid = Grid(x_min=-mem_cells[0] * grid.res / 2, x_max=mem_cells[0] * grid.res / 2,
                             y_min=-mem_cells[1] * grid.res / 2, y_max=mem_cells[1] * grid.res / 2, res=grid.res)
        self.v_static, self.mem_tau, self.w_max = v_static, mem_tau, w_max
        self.head = VelocityHead(c_in, head_hidden)
        self.relevance = nn.Conv2d(c_in, 1, 1)
        nn.init.constant_(self.relevance.bias, 2.0)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_src)))
        self.fuse = nn.Sequential(nn.Conv2d(3 * c_in + 3, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
                                  nn.ReLU(True))
        self.register_buffer("ego_pts", grid_points(grid))

    def init_state(self, B, device=None, dtype=torch.float32):
        H, Wd = self.mem_grid.nx, self.mem_grid.ny
        return {"M": torch.zeros(B, self.c_in, H, Wd, device=device, dtype=dtype),
                "W": torch.zeros(B, 1, H, Wd, device=device, dtype=dtype),
                "pose": torch.zeros(B, 3, device=device, dtype=dtype)}

    @staticmethod
    def detach_state(state):
        return {k: v.detach() for k, v in state.items()}

    def aggregate(self, out_w, mass_w, ages):
        """out_w (B, K, C, X, Y), mass_w (B, K, 1, X, Y), ages (K,) or per sample (B, K) seconds -> agg (B, C, X, Y),
        summed mass, mean age."""
        B, K, C, X, Y = out_w.shape
        rel = torch.sigmoid(self.relevance(out_w.reshape(B * K, C, X, Y))).reshape(B, K, 1, X, Y)
        ages = ages.to(out_w)
        age = ages.view(1, K, 1, 1, 1) if ages.dim() == 1 else ages.view(B, K, 1, 1, 1)
        m = mass_w.clamp(max=1.0)
        a = m * torch.exp(-age / torch.exp(self.log_tau)) * rel
        den = a.sum(1)
        return (a * out_w).sum(1) / (den + 1e-6), m.sum(1), (a * age).sum(1) / (den + 1e-6)

    def forward(self, cur, past=(), state=None, step=None, obs_mass=None):
        """cur (B, C, X, Y) current planes; past: list of (bev, steps) original past frames with their per-frame ego steps
        [(dt, v, omega), ...] to the current time (as splat_window); state: the dict of the previous call or None; step:
        (dt, v, omega) from the previous current frame to this one (moves the memory pose), or the chain of such triples
        (advanced one after the other, the decay time is their sum); obs_mass (B, 1, X, Y): where the sensor observed.
        Returns out (B, c_out, X, Y), the new state and aux (v_hat, static, mass_sum, age_mean, mem_read, mem_conf, agg)."""
        B, C, X, Y = cur.shape
        state = state or self.init_state(B, cur.device, cur.dtype)
        pose = advance_chain(state["pose"], step)
        M, W, pose = recentre(state["M"], state["W"], pose, self.grid.res)
        v_hat = self.head(cur)
        static = torch.exp(-v_hat.pow(2).sum(1, keepdim=True) / self.v_static ** 2)
        if len(past):
            out_w, mass_w = splat_window(list(past), self.grid, head=self.head, mass_gate=True)
            # step values may be python scalars or per-sample (B,) tensors (nuScenes batches): ages (B, K)
            ages = torch.stack([sum(_col(s[0], cur) for s in steps) * cur.new_ones(B) for _, steps in past], 1).abs()
            agg, msum, age_mean = self.aggregate(out_w, mass_w, ages)
        else:
            agg, msum, age_mean = torch.zeros_like(cur), cur.new_zeros(B, 1, X, Y), cur.new_zeros(B, 1, X, Y)
        m_read, conf = read_memory(M, W, pose, self.ego_pts, X, Y, self.mem_grid)
        out = self.fuse(torch.cat([cur, agg, msum, age_mean, m_read, (conf / self.w_max).clamp(max=1.0)], 1))
        om = obs_mass if obs_mass is not None else cur.new_ones(B, 1, X, Y)
        feats = torch.cat([cur.reshape(B, C, -1), static.reshape(B, 1, -1), om.reshape(B, 1, -1)], 1)
        sp, mass = splat(feats, ego_to_memory(pose, self.ego_pts), self.mem_grid)
        w_new = mass.clamp(max=1.0) * sp[:, C:C + 1] * sp[:, C + 1:C + 2]
        dt = sum((_col(s[0], cur).abs() for s in as_chain(step)), cur.new_zeros(B))              # scalar or (B,) per step, summed over a chain
        Wd = W * torch.exp(-dt / self.mem_tau).view(B, 1, 1, 1)
        Wt = Wd + w_new
        M = (M * Wd + sp[:, :C] * w_new) / (Wt + 1e-6)
        W = Wt.clamp(max=self.w_max)
        aux = {"v_hat": v_hat, "static": static, "mass_sum": msum, "age_mean": age_mean, "mem_read": m_read,
               "mem_conf": conf, "agg": agg}
        return out, {"M": M, "W": W, "pose": pose}, aux


# ----------------------------------------------------------------------------------------------- self-test
def pattern(x, y):
    """A static world: two lane lines along x, cross lines every 12 m, one blob (values 0..1)."""
    lines = torch.exp(-((y - 3.5) / 0.4) ** 2) + torch.exp(-((y + 3.5) / 0.4) ** 2)
    cross = sum(torch.exp(-((x - xc) / 0.4) ** 2) for xc in (10.0, 22.0, 34.0, 46.0)) * (y.abs() < 6).to(x.dtype)
    blob = torch.exp(-(((x - 30.0) ** 2 + (y + 1.0) ** 2) / 1.0))
    return (lines + cross + blob).clamp(max=1.0)


def _observe(pose, pts, x_fov, extra=None):
    """The ego planes at `pose`: channel 0 = pattern at the cell centres, channel 1 = half of it; obs_mass = x >= x_fov."""
    w = ego_to_memory(pose, pts)
    val = pattern(w[..., 0], w[..., 1])
    if extra is not None:
        val = (val + extra(w)).clamp(max=1.0)
    cur = torch.stack([val, 0.5 * val], 1)
    return cur, w, (pts[:, 0] >= x_fov).to(val.dtype)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)


def _drive(mod, grid, v, omega, frames, x_fov=8.0, dt=1 / 12, moving=None, ignore_motion=False):
    """Drive the module over a static world (+ optional moving blob); returns per-frame (corr, coverage, ghost) of the
    memory read in the cells the sensor does not observe now (x < x_fov - 1 m)."""
    B = 1
    pts = grid_points(grid)
    X, Y = grid.nx, grid.ny
    state, pose_true = None, torch.zeros(B, 3)
    if ignore_motion:
        mod.v_static = 1e6
    log = []
    for t in range(frames):
        step = (dt, v, omega) if t else None
        if t:
            pose_true = advance_pose(pose_true, *step)
        extra = None
        if moving is not None:
            extra = lambda w, t=t: torch.exp(-(((w[..., 0] - 16.0) ** 2 + (w[..., 1] - (-6.0 + moving * t * dt)) ** 2) / 0.8))
        cur, w, om = _observe(pose_true, pts, x_fov, extra)
        cur3 = cur
        if moving is not None:                         # channels 1, 2 carry the blob's velocity (vx = 0, vy) on its footprint
            on_blob = (extra(w) > 0.02).to(cur.dtype)
            cur3 = torch.stack([cur[:, 0], torch.zeros_like(cur[:, 0]), moving * on_blob], 1)
        with torch.no_grad():
            out, state, aux = mod(cur3.view(B, -1, X, Y), (), state, step, om.view(B, 1, X, Y))
        un = (pts[:, 0] < x_fov - 1.0).view(1, X, Y)
        conf = aux["mem_conf"][:, 0] / mod.w_max
        valid = un & (conf > 0.3)
        truth = pattern(w[..., 0], w[..., 1]).view(1, X, Y)
        read = aux["mem_read"][:, 0]
        corr = _ncc(read[valid], truth[valid]) if valid.sum() > 20 else float("nan")
        cover = (valid.sum() / un.sum()).item()
        # moving test: the memory read along the blob's path (y < current y - 2 m, x near 16 m) should equal the static world
        ghost = float("nan")
        if moving is not None:
            path = ((w[..., 0] - 16.0).abs() < 1.2) & (w[..., 1] < (-6.0 + moving * t * dt) - 2.0) & (w[..., 1] > -7.0)
            path = path.view(1, X, Y) & (conf > 0.05)
            ghost = (read - truth)[path].abs().mean().item() if path.any() else 0.0   # deviation from the STATIC world
        log.append((float(corr), cover, ghost))
    return log


def _selftest():
    torch.manual_seed(0)
    grid = Grid(x_min=0.0, x_max=24.0, y_min=-8.0, y_max=8.0, res=0.5)           # 48 x 32 cells for the tests
    X, Y = grid.nx, grid.ny

    # 1 pose + re-centring: lossless (reads before / after are identical), content moves by whole cells
    mod = TemporalFusion(2, 8, grid, mem_cells=(96, 64)).eval()
    st = mod.init_state(1)
    st["M"], st["W"] = torch.rand_like(st["M"]), torch.rand_like(st["W"]) + 0.5
    pose = torch.tensor([[3.2, -1.7, 0.4]])
    r0 = read_memory(st["M"], st["W"], pose, mod.ego_pts, X, Y, mod.mem_grid)
    M2, W2, pose2 = recentre(st["M"], st["W"], pose, grid.res)
    r1 = read_memory(M2, W2, pose2, mod.ego_pts, X, Y, mod.mem_grid)
    inside = ((r0[1] - r1[1]).abs() < 1e-6) & (r0[1] > 1e-3)            # same bilinear support: no dropped strip involved
    assert inside.sum() > 200, "enough cells to compare"
    assert torch.allclose(r0[0][inside.expand_as(r0[0])], r1[0][inside.expand_as(r1[0])], atol=1e-5), "re-centring lossless"
    assert abs(float(pose2[0, 0])) <= grid.res / 2 + 1e-6 and abs(float(pose2[0, 1])) <= grid.res / 2 + 1e-6

    # 2 aggregation: the fresher source dominates (tau 0.3 s), mass 0 sources do not count
    mod.log_tau.data = torch.tensor(math.log(0.3))
    ow = torch.stack([torch.ones(1, 2, X, Y), 2 * torch.ones(1, 2, X, Y)], 1)       # (B, K=2, C, X, Y)
    mw = torch.ones(1, 2, 1, X, Y)
    with torch.no_grad():
        agg, msum, age = mod.aggregate(ow, mw, torch.tensor([0.1, 0.8]))
        assert agg.mean() < 1.2 and abs(float(msum.mean()) - 2.0) < 1e-6 and age.mean() < 0.25, "fresh source dominates"
        mw0 = mw.clone()
        mw0[:, 0] = 0
        agg0, msum0, _ = mod.aggregate(ow, mw0, torch.tensor([0.1, 0.8]))
        assert abs(float(agg0.mean()) - 2.0) < 1e-3 and abs(float(msum0.mean()) - 1.0) < 1e-6, "empty source ignored"

    # 3 static world, straight ego motion (10 m/s, 36 frames): the memory fills the cells the sensor does not see now,
    #   matches the world pattern, and does not degrade with time (no compounding blur)
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).eval()
    log = _drive(mod, grid, 10.0, 0.0, 36)
    c14, c36 = log[14][0], log[35][0]
    assert c14 > 0.9 and c36 > 0.9, f"memory read vs world pattern: {c14:.3f}, {c36:.3f}"
    assert abs(c14 - c36) < 0.04 and log[35][1] > 0.6, "no compounding, coverage of the unobserved cells"

    # 4 turning ego motion (8 m/s, 0.5 rad/s, 36 frames = 1.5 rad)
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).eval()
    logt = _drive(mod, grid, 8.0, 0.5, 36)
    # (after about 2.3 s of the sharp turn the unobserved cells hold no pattern any more: the correlation is only
    #  defined while they do, so the frames 8-24 are judged)
    assert min(c for c, _, _ in logt[8:25]) > 0.88 and logt[20][1] > 0.3, f"turning: {logt[8:25]}"

    # 5 a moving object (6 m/s sideways) does not leave a ghost trail in the memory when the velocity head reports it
    #   (static weight ~ 0); without the velocity (everything static) it does
    def moving_head_module():
        # w_max 8 here: the test shows the ghost mechanism; with the default 2 the memory forgets faster and the ghost
        # without velocity is smaller (0.063 instead of > 0.1, 2026-10-26)
        m = TemporalFusion(3, 8, grid, mem_cells=(128, 96), w_max=8.0).eval()
        w1 = torch.zeros_like(m.head.net[0].weight)
        w1[0, 1], w1[1, 1], w1[2, 2], w1[3, 2] = 1, -1, 1, -1                       # hidden units pass vx, vy (ReLU split)
        m.head.net[0].weight.data = w1
        m.head.net[0].bias.data.zero_()
        w2 = torch.zeros_like(m.head.net[2].weight)
        w2[0, 0], w2[0, 1], w2[1, 2], w2[1, 3] = 1, -1, 1, -1
        m.head.net[2].weight.data, m.head.net[2].bias.data = w2, torch.zeros(2)
        return m
    ghost_h = max(g for _, _, g in _drive(moving_head_module(), grid, 0.0, 0.0, 30, moving=6.0)[10:])
    ghost_o = max(g for _, _, g in _drive(moving_head_module(), grid, 0.0, 0.0, 30, moving=6.0, ignore_motion=True)[10:])
    assert ghost_h < 0.02 and ghost_o > 0.1 and ghost_o > 20 * ghost_h, f"ghost with velocity {ghost_h:.3f}, without {ghost_o:.3f}"

    # 6 window sources: under pure ego motion the aggregated past frames agree with the current frame where they landed
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).eval()
    pts = grid_points(grid)
    dt, v = 1 / 12, 10.0
    p0 = torch.zeros(1, 3)
    p1 = advance_pose(p0, dt, v, 0.0)
    p2 = advance_pose(p1, dt, v, 0.0)
    c0, _, _ = _observe(p0, pts, 0.0)
    c1, _, _ = _observe(p1, pts, 0.0)
    c2, _, _ = _observe(p2, pts, 0.0)
    past = [(c0.view(1, 2, X, Y), [(dt, v, 0.0), (dt, v, 0.0)]), (c1.view(1, 2, X, Y), [(dt, v, 0.0)])]
    with torch.no_grad():
        out, _, aux = mod(c2.view(1, 2, X, Y), past, None, None)
    ok = aux["mass_sum"][0, 0] > 1.9
    err = (aux["agg"][0, 0] - c2.view(1, 2, X, Y)[0, 0])[ok].abs().mean().item()
    assert ok.sum() > 100 and err < 0.12, f"window sources vs current: err {err:.3f}, {int(ok.sum())} cells"
    assert out.shape == (1, 8, X, Y) and torch.isfinite(out).all()

    # 7 gradients: the head (through the window positions and the memory weights), the relevance gate, tau and the fusion
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).train()
    st, pose_t = None, torch.zeros(1, 3)
    for t in range(3):
        step = (dt, v, 0.0) if t else None
        if t:
            pose_t = advance_pose(pose_t, *step)
        cur, _, om = _observe(pose_t, pts, 4.0)
        pw = [(c0.view(1, 2, X, Y), [(dt, v, 0.0)] * (t + 1))] if t else []
        out, st, _ = mod(cur.view(1, 2, X, Y), pw, st, step, om.view(1, 1, X, Y))
    out.mean().backward()
    assert mod.head.net[2].weight.grad is not None and mod.head.net[2].weight.grad.abs().sum() > 0, "head gradient"
    assert mod.relevance.weight.grad is not None and mod.log_tau.grad is not None
    assert mod.fuse[0].weight.grad is not None

    # 10 per-sample steps (nuScenes batches: different ego speed, yaw rate and dt per sample as (B,) tensors): one batched call
    #    equals one call per sample (ages, memory decay and the pose all follow the sample)
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).eval()
    vs, oms, dts = torch.tensor([6.0, 12.0]), torch.tensor([0.0, 0.4]), torch.tensor([1 / 12, 1 / 6])

    def drive_batch(idx):
        n = len(idx)
        v_, om_, dt_ = vs[idx], oms[idx], dts[idx]
        st_, pose_, origin, out_ = None, torch.zeros(n, 3), [], None
        for t in range(4):
            step_ = (dt_, v_, om_) if t else None
            if t:
                pose_ = advance_pose(pose_, *step_)
            cur_, _, mask = _observe(pose_, pts, 4.0)
            cur_ = cur_.view(n, 2, X, Y)
            pw_ = [(origin[j], [(dt_, v_, om_)] * (t - j)) for j in range(t)][-2:]
            with torch.no_grad():
                out_, st_, _ = mod(cur_, pw_, st_, step_, mask.view(1, 1, X, Y).expand(n, 1, X, Y))
            origin.append(cur_)
        return out_, st_

    out_b, st_b = drive_batch([0, 1])
    for i in (0, 1):
        out_i, st_i = drive_batch([i])
        assert torch.allclose(out_b[i], out_i[0], atol=1e-5), f"batched vs single sample {i}"
        assert torch.allclose(st_b["W"][i], st_i["W"][0], atol=1e-5), f"memory confidence of sample {i}"
    assert not torch.allclose(st_b["W"][0], st_b["W"][1]), "the samples really differ"

    # 8 empty inputs: no past, no memory content -> finite output, zero mass channels in the input of the conv
    mod = TemporalFusion(2, 8, grid, mem_cells=(96, 64)).eval()
    with torch.no_grad():
        out, st, aux = mod(torch.rand(2, 2, X, Y))
    assert torch.isfinite(out).all() and aux["mass_sum"].abs().sum() == 0
    print(f"temporal_fusion selftest ok: lossless re-centring, fresh source dominates, memory vs world pattern "
          f"(straight corr {c14:.3f} -> {c36:.3f}, coverage {log[35][1]:.2f}; turning corr >= "
          f"{min(c for c, _, _ in logt[8:25]):.3f} over frames 8-24), no ghost trail with the velocity head "
          f"({ghost_h:.3f} vs {ghost_o:.3f} without), window sources, gradients, "
          f"per-sample steps = one batched call, empty inputs")

    try:
        from torch.utils.flop_counter import FlopCounterMode
        big = TemporalFusion(64, 64, Grid(), mem_cells=(256, 256)).eval()
        cur = torch.rand(1, 64, 120, 104)
        pw = [(torch.rand(1, 64, 120, 104), [(1 / 12, 10.0, 0.0)] * (k + 1)) for k in range(3)]
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            big(cur, pw, None, None)
        print(f"  at 64 channels, 120 x 104, K = 3: {fc.get_total_flops() / 1e9:.1f} GFLOPs (convs + matmuls), "
              f"{sum(p.numel() for p in big.parameters()) / 1e3:.0f} k parameters")
    except Exception as e:                                        # the size line is informative only
        print("  (size line skipped:", e, ")")


if __name__ == "__main__":
    _selftest()

"""Per-branch temporal fusion (the owner 2026-10-25): the state update of one branch (camera
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

ONE code path for camera and radar (the owner's rule): the two branches differ only in the data they pass, not in the module.
Conventions as net/model/splat.py: x forward, y left, maps (B, C, X, Y).
Tests: tests/test_temporal_fusion.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_fusion.net.model.lift_pd import Grid
from bev_fusion.net.model.splat import VelocityHead, grid_points, splat, splat_window


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
    for b in range(M.shape[0]):
        sx, sy = int(shift[b, 0]), int(shift[b, 1])
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

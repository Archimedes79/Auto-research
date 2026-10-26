"""Temporal propagation: one robust building block that moves a BEV feature map (or a set of points with features) from
its source time to a target time - usable for the camera history, the radar history, the radar -> camera hop and the
prediction at any dt (plan.md sections 7, 8.2, 8.3). Speed can be learned end to end.

Per cell (or point) p with feature F(p), per sample the ego steps S = (dt_i, v_i, omega_i) from source to target:
  hypotheses   v_m = f_m(F(p)), m = 1..M learned, plus optionally v_0 = 0 (static); score s_m(F(p))
  evidence     optional Doppler per cell: J detections with unit ray direction u_j and radial velocity d_j (explicit
               layers, used only here): r_jm = wrap(v_m . u_j - d_j), wrap = into the unambiguous interval
               [-v_alias, v_alias); s_m += lambda * sum_j valid_j * (-r_jm^2 / (2 sigma_d^2))
  weights      pi_m = softmax_m(s_m)
  positions    p'_m = chain(p, S, v_m)     the exact geometry of training/splat.py (coordinates chained per step,
               the object velocity rotated with the ego; translation first order in omega, rotation exact)
  splat        out(q) = sum_{p,m} pi_m k(q - p'_m) F(p) / sum_{p,m} pi_m k(q - p'_m) (* min(mass, 1) if mass_gate)
               mass(q) = sum_{p,m} pi_m k(q - p'_m)  (0 = unknown)
               k: bilinear by default (sigma = 0: exact identity at dt = 0, exact shifts, no blur); on request a
               Gaussian of width sigma = sqrt(sigma0^2 + (sigma_v |dt|)^2) cells within +-radius cells, normalised per
               source - for long-range gradients early in training (anneal sigma0 to 0) or honestly blurrier forecasts
Why this is not destructive for learning speed end to end: (1) a Gaussian kernel (on request) passes a position
gradient over its width (the bilinear kernel only within one cell), (2) the softmax over hypotheses passes a gradient to the CHOICE
between far-apart velocities (non-local credit), (3) Doppler evidence pins the radial part; (4) the static
hypothesis keeps the static world exact; (5) nothing is lost on the way - every source keeps its mass, unknown cells
are marked by mass 0. Velocity labels, where available, can still supervise v_mean (aux loss).
Conventions as training/splat.py: x forward, y left, maps (B, C, X, Y), cell i centre x_min + (i + 0.5) res.
Basis: softmax splatting (Niklaus & Liu, CVPR 2020, from memory), multiple-hypothesis / winner-takes-all prediction
(e.g. MCL; from memory), CRT-Fusion's velocity shift and FIERY's ego warp (literature/code/).
Run the self-test: python training/temporal.py; the gate tests: tests/test_temporal.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lift_pd import Grid
    from .splat import grid_points, splat_positions
except ImportError:                                   # run as a script
    from lift_pd import Grid
    from splat import grid_points, splat_positions


def wrap(x, v_alias):
    """into [-v_alias, v_alias): a radar measures radial velocity only modulo 2 v_alias (no wrap if v_alias is None)."""
    if v_alias is None:
        return x
    return torch.remainder(x + v_alias, 2 * v_alias) - v_alias


def steps_tensor(steps, B, device=None, dtype=torch.float32):
    """steps -> (B, S, 3) tensor of (dt, v, omega). Accepts a tensor (B, S, 3), one list of (dt, v, omega) for all
    samples, or a list of B such lists (different lengths are padded with dt = 0, which is the identity step)."""
    if torch.is_tensor(steps):
        return steps.to(device=device, dtype=dtype)
    if len(steps) == 0:
        return torch.zeros(B, 1, 3, device=device, dtype=dtype)
    if not isinstance(steps[0][0], (list, tuple)):
        steps = [steps] * B
    S = max(1, max(len(s) for s in steps))
    out = torch.zeros(B, S, 3, device=device, dtype=dtype)
    for b, s in enumerate(steps):
        if s:
            out[b, :len(s)] = torch.tensor(s, device=device, dtype=dtype)
    return out


def chain(p, steps, v):
    """p (B, N, 2) source positions, steps (B, S, 3), v (B, N, 2) velocity in the source axes -> target positions.
    Coordinates chained step by step (training/splat.py splat_positions), the velocity vector rotated with the ego."""
    for i in range(steps.shape[1]):
        dt, ve, w = steps[:, i, 0], steps[:, i, 1], steps[:, i, 2]
        p = splat_positions(p, dt, ve, w, v)
        th = (w * dt).view(-1, 1)
        c, s = torch.cos(th), torch.sin(th)
        v = torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)
    return p


def gaussian_splat(feat, pos, weight, grid, sigma, radius, eps=1e-6, mass_gate=True, importance=None):
    """feat (B, C, N), pos (B, N, 2) metres, weight (B, N) >= 0 (e.g. hypothesis probabilities), sigma (B,) or scalar in
    cells -> out (B, C, X, Y) normalised, mass (B, 1, X, Y). Every source spreads its weight over the (2 radius + 1)^2
    cells around it with a Gaussian normalised to sum 1 (mass conserving, also outside the grid it is just lost).
    importance (B, N) > 0 or None: softmax splatting (Niklaus & Liu 2003.05534) - the feature is the importance-
    weighted mean of the sources landing in a cell, sum(w imp f) / sum(w imp), so an object can win over the empty
    cells landing on it; the mass (known / unknown) stays sum(w). None = importance 1 (the plain normalised splat)."""
    B, C, N = feat.shape
    X, Y = grid.nx, grid.ny
    fx = (pos[..., 0] - grid.x_min) / grid.res - 0.5
    fy = (pos[..., 1] - grid.y_min) / grid.res - 0.5
    sig = torch.as_tensor(sigma, dtype=feat.dtype, device=feat.device).reshape(-1, 1).expand(B, 1)
    if float(sig.max()) <= 0 or radius == 0:
        # bilinear (the default): exact identity for dt = 0 and exact sub-cell shifts, no blur
        bx, by = torch.floor(fx).detach(), torch.floor(fy).detach()
        offs = torch.tensor([0.0, 1.0], device=feat.device, dtype=feat.dtype)
        ox, oy = torch.meshgrid(offs, offs, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        cx, cy = bx.unsqueeze(-1) + ox, by.unsqueeze(-1) + oy
        k = (1 - (cx - fx.unsqueeze(-1)).abs()) * (1 - (cy - fy.unsqueeze(-1)).abs())
    else:
        # Gaussian of width sigma (cells), normalised per source: longer-range position gradients, honest blur
        bx, by = torch.round(fx).detach(), torch.round(fy).detach()
        offs = torch.arange(-radius, radius + 1, device=feat.device, dtype=feat.dtype)
        ox, oy = torch.meshgrid(offs, offs, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)                               # (T,)
        cx, cy = bx.unsqueeze(-1) + ox, by.unsqueeze(-1) + oy                 # (B, N, T) cell indices
        s2 = sig.clamp(min=0.3).unsqueeze(-1) ** 2
        k = torch.exp(-((cx - fx.unsqueeze(-1)) ** 2 + (cy - fy.unsqueeze(-1)) ** 2) / (2 * s2))
        k = k / k.sum(-1, keepdim=True).clamp(min=eps)                        # per source sum 1
    ok = (cx >= 0) & (cx < X) & (cy >= 0) & (cy < Y)
    w = k * weight.unsqueeze(-1) * ok
    idx = (cx.clamp(0, X - 1) * Y + cy.clamp(0, Y - 1)).long() + (torch.arange(B, device=feat.device) * X * Y).view(B, 1, 1)
    flat, wf = idx.reshape(-1), w.reshape(-1)
    T = ox.numel()
    src = feat.unsqueeze(-1).expand(B, C, N, T).permute(1, 0, 2, 3).reshape(C, -1)
    mass = feat.new_zeros(B * X * Y).index_add(0, flat, wf)
    if importance is not None:
        wf = (w * importance.unsqueeze(-1)).reshape(-1)
        den = feat.new_zeros(B * X * Y).index_add(0, flat, wf)
    else:
        den = mass
    num = feat.new_zeros(C, B * X * Y).index_add(1, flat, src * wf)
    out = (num / den.clamp(min=eps)).view(C, B, X, Y).permute(1, 0, 2, 3)
    mass = mass.view(B, 1, X, Y)
    if mass_gate:
        out = out * mass.clamp(max=1.0)
    return out, mass


class TemporalPropagation(nn.Module):
    """Move features from their source time to a target time. Grid mode: forward(feat (B, C, X, Y), steps, ...);
    point mode: forward_points(feat (B, C, N), pos (B, N, 2), steps, ...). Returns a dict: feat, mass, v_hyp
    (B, M, 2, ...), pi (B, M, ...), v_mean (B, 2, ...)."""

    def __init__(self, cin, grid=None, n_hyp=4, static_hyp=True, hidden=64, v_max=40.0, init_speed=5.0,
                 sigma0=0.0, sigma_v=0.0, radius=2, doppler_sigma=0.3, v_alias=None, mass_gate=True,
                 importance=False, temperature=1.0):
        super().__init__()
        self.importance, self.temperature = importance, temperature   # temperature < 1 sharpens pi (anneal it)
        self.grid = grid or Grid()
        self.n_hyp, self.static_hyp, self.v_max = n_hyp, static_hyp, v_max
        self.sigma0, self.sigma_v, self.radius = sigma0, sigma_v, radius
        self.doppler_sigma, self.v_alias, self.mass_gate = doppler_sigma, v_alias, mass_gate
        M = n_hyp + int(static_hyp)
        self.net = nn.Sequential(nn.Conv2d(cin, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, 2 * n_hyp + M + int(importance), 1))
        last = self.net[4]
        nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():                         # learned hypotheses start as a fan of speeds and directions
            last.bias.zero_()
            for m in range(n_hyp):
                a = 2 * math.pi * m / max(1, n_hyp)
                last.bias[2 * m] = math.atanh(min(0.99, init_speed * math.cos(a) / v_max)) * v_max
                last.bias[2 * m + 1] = math.atanh(min(0.99, init_speed * math.sin(a) / v_max)) * v_max
        self.log_lambda = nn.Parameter(torch.zeros(()))                       # weight of the Doppler evidence

    def hypotheses(self, feat):
        """feat (B, C, N) -> v (B, M, 2, N) in m/s, logits (B, M, N)."""
        B, C, N = feat.shape
        raw = self.net(feat.unsqueeze(-1))[..., 0]                             # (B, 2 n + M, N)
        v = self.v_max * torch.tanh(raw[:, :2 * self.n_hyp] / self.v_max).view(B, self.n_hyp, 2, N)
        if self.static_hyp:
            v = torch.cat([v.new_zeros(B, 1, 2, N), v], 1)                  # (n_hyp may be 0: ego only)
        M = self.n_hyp + int(self.static_hyp)
        return v, raw[:, 2 * self.n_hyp:2 * self.n_hyp + M]

    def importance_of(self, feat):
        """feat (B, C, N) -> softmax-splatting importance exp(z) (B, N), z bounded to +-5; 1 at initialisation."""
        z = self.net(feat.unsqueeze(-1))[:, -1, :, 0]
        return torch.exp(5.0 * torch.tanh(z / 5.0))

    def doppler_loglik(self, v, doppler):
        """v (B, M, 2, N); doppler: u (B, J, 2, N) unit rays, d (B, J, N) radial velocities, valid (B, J, N)
        -> (B, M, N) sum_j valid * (-r^2 / (2 sigma^2)), r = wrap(v . u - d)."""
        pred = (v.unsqueeze(2) * doppler["u"].unsqueeze(1)).sum(3)            # (B, M, J, N)
        r = wrap(pred - doppler["d"].unsqueeze(1), self.v_alias)
        return (-(r ** 2) / (2 * self.doppler_sigma ** 2) * doppler["valid"].unsqueeze(1)).sum(2)

    def forward_points(self, feat, pos, steps, doppler=None, dt_total=None):
        B, C, N = feat.shape
        S = steps_tensor(steps, B, feat.device, feat.dtype)
        v, logits = self.hypotheses(feat)
        if doppler is not None:
            logits = logits + self.log_lambda.exp() * self.doppler_loglik(v, doppler)
        pi = (logits / self.temperature).softmax(1)                            # (B, M, N)
        entropy = -(pi * pi.clamp(min=1e-12).log()).sum(1)                     # (B, N): hedging per source
        M = v.shape[1]
        p = pos.unsqueeze(1).expand(B, M, N, 2).reshape(B * M, N, 2)
        vv = v.permute(0, 1, 3, 2).reshape(B * M, N, 2)
        target = chain(p, S.repeat_interleave(M, 0), vv).view(B, M * N, 2)
        dt = S[:, :, 0].sum(1) if dt_total is None else torch.as_tensor(dt_total, dtype=feat.dtype, device=feat.device)
        sigma = torch.sqrt(self.sigma0 ** 2 + (self.sigma_v * dt.abs() / self.grid.res) ** 2).reshape(-1)
        imp = self.importance_of(feat).repeat(1, M) if self.importance else None
        out, mass = gaussian_splat(feat.repeat(1, 1, M), target, pi.reshape(B, M * N), self.grid,
                                   sigma.expand(B) if sigma.numel() == 1 else sigma, self.radius,
                                   mass_gate=self.mass_gate, importance=imp)
        v_mean = (pi.unsqueeze(2) * v).sum(1)                                   # (B, 2, N)
        v_best = v.gather(1, pi.argmax(1).view(B, 1, 1, N).expand(B, 1, 2, N))[:, 0]   # (B, 2, N), for reading only
        return {"feat": out, "mass": mass, "v_hyp": v, "pi": pi, "v_mean": v_mean, "v_best": v_best,
                "entropy": entropy, "target": target}

    def forward(self, feat, steps, doppler=None):
        B, C, X, Y = feat.shape
        pos = grid_points(self.grid, feat.device, feat.dtype).unsqueeze(0).expand(B, -1, -1)
        dop = None
        if doppler is not None:
            dop = {"u": doppler["u"].flatten(3), "d": doppler["d"].flatten(2), "valid": doppler["valid"].flatten(2)}
        r = self.forward_points(feat.flatten(2), pos, steps, dop)
        M = r["v_hyp"].shape[1]
        return {"feat": r["feat"], "mass": r["mass"], "v_hyp": r["v_hyp"].view(B, M, 2, X, Y),
                "pi": r["pi"].view(B, M, X, Y), "v_mean": r["v_mean"].view(B, 2, X, Y),
                "v_best": r["v_best"].view(B, 2, X, Y), "entropy": r["entropy"].view(B, X, Y)}

    def set_velocity(self, v_xy, static_weight=-20.0):
        """Test helper: every learned hypothesis returns v_xy (m/s) for any input; the static one is suppressed."""
        last = self.net[4]
        with torch.no_grad():
            last.weight.zero_()
            for m in range(self.n_hyp):
                last.bias[2 * m] = math.atanh(v_xy[0] / self.v_max) * self.v_max
                last.bias[2 * m + 1] = math.atanh(v_xy[1] / self.v_max) * self.v_max
            last.bias[2 * self.n_hyp:] = 0.0
            if self.static_hyp:
                last.bias[2 * self.n_hyp] = static_weight


def doppler_update(v_p, P_p, u, d, valid, sigma_d=0.3, v_alias=None):
    """Per-cell Kalman measurement update of a velocity with radial (Doppler) velocities, in information form (plan.md
    7.3, step 2; closed form, differentiable). v_p (B, 2, N) prior velocity, P_p (B, 2, 2, N) prior covariance; per
    cell J detections: u (B, J, 2, N) unit rays, d (B, J, N) radial velocities, valid (B, J, N) in [0, 1].
    The measurement d_j = u_j . v + noise(sigma_d); with aliasing, d_j is first unwrapped to the alias closest to the
    prior (u_j . v_p). A = sum valid u u^T / sigma^2, b = sum valid u d / sigma^2;
    P = (P_p^-1 + A)^-1, v = P (P_p^-1 v_p + b). The radial part comes from the radar, the tangential part keeps the
    prior and its variance (a single ray says nothing across it). -> v (B, 2, N), P (B, 2, 2, N)."""
    pred = (u * v_p.unsqueeze(1)).sum(2)                                          # (B, J, N)
    if v_alias is not None:
        d = d + 2 * v_alias * torch.round((pred - d) / (2 * v_alias)).detach()
    w = valid / sigma_d ** 2
    A = (w.unsqueeze(2).unsqueeze(2) * u.unsqueeze(3) * u.unsqueeze(2)).sum(1)  # (B, 2, 2, N)
    b = (w.unsqueeze(2) * u * d.unsqueeze(2)).sum(1)                              # (B, 2, N)

    def inv(M):                                                                   # closed-form 2 x 2 inverse per cell
        a, b_, c, e = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        det = a * e - b_ * c
        return torch.stack([torch.stack([e, -b_], 1), torch.stack([-c, a], 1)], 1) / det.unsqueeze(1).unsqueeze(1)
    Pi = inv(P_p)
    P = inv(Pi + A)
    rhs = (Pi * v_p.unsqueeze(1)).sum(2) + b                                      # P_p^-1 v_p + b
    return (P * rhs.unsqueeze(1)).sum(2), P


def landing(steps, v, grid):
    """steps (B, S, 3), v (B, 2, X, Y) one velocity per cell (m/s, source axes) -> landing points (B, X*Y, 2) of
    the cell centres in the target frame."""
    B = v.shape[0]
    p = grid_points(grid, v.device, v.dtype).unsqueeze(0).expand(B, -1, -1)
    return chain(p, steps, v.flatten(2).transpose(1, 2))


def sample_at(feat, pos, grid):
    """feat (B, C, X, Y), pos (B, N, 2) metres -> (B, C, N) bilinear samples (zero outside the grid)."""
    u = 2 * (pos[..., 1] - grid.y_min) / (grid.ny * grid.res) - 1                  # W axis = y
    w = 2 * (pos[..., 0] - grid.x_min) / (grid.nx * grid.res) - 1                  # H axis = x
    return F.grid_sample(feat, torch.stack([u, w], -1).unsqueeze(1), mode="bilinear", padding_mode="zeros",
                         align_corners=False)[:, :, 0]


def propagate(feat, steps, v, grid, mass_gate=True, importance=None):
    """move a map with one velocity per cell: feat (B, C, X, Y), v (B, 2, X, Y), importance (B, X, Y) or None
    (softmax splatting) -> splat (B, C, X, Y), mass."""
    B, C, X, Y = feat.shape
    S = steps_tensor(steps, B, feat.device, feat.dtype)
    return gaussian_splat(feat.flatten(2), landing(S, v, grid), feat.new_ones(B, X * Y), grid, 0.0, 2,
                          mass_gate=mass_gate, importance=None if importance is None else importance.flatten(1))


class FlowRefine(nn.Module):
    """Iterative refinement of a per-cell velocity field (a RAFT-like update operator with shared weights). Per
    iteration: the source cells land at p' under the current v; the current frame is looked up at (2 r + 1)^2 offsets
    of one cell around p' and correlated with the source cell's feature (learned 1x1 projections); a small conv net
    turns [correlation, source feature, v] into a displacement step d (metres, target axes), and dv = R(theta) d / dt
    (theta = total ego yaw, dt = total time), so the operator does not depend on dt. The step is bounded by the lookup
    window, |d| <= step_max = radius * res per component (tanh): without the bound a warp-only loss learns to throw
    objects out of the grid (nothing predicted is cheaper than a blob in the wrong place; EPE 72 m/s in the first
    test). v is detached between iterations (as in RAFT); the last layer starts at zero, so v0 is kept until something
    is learned. Grid mode only."""

    def __init__(self, cin, grid, radius=3, proj=16, hidden=64, v_scale=10.0, step_max=None):
        super().__init__()
        self.grid, self.v_scale = grid, v_scale
        self.step_max = radius * grid.res if step_max is None else step_max
        self.f, self.g = nn.Conv2d(cin, proj, 1), nn.Conv2d(cin, proj, 1)
        o = torch.arange(-radius, radius + 1, dtype=torch.float32) * grid.res
        ox, oy = torch.meshgrid(o, o, indexing="ij")
        self.register_buffer("offs", torch.stack([ox.reshape(-1), oy.reshape(-1)], -1), persistent=False)
        T = self.offs.shape[0]
        self.net = nn.Sequential(nn.Conv2d(T + proj + 2, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correlation(self, fs, gc, pos):
        """fs (B, P, X, Y) source projection, gc (B, P, X, Y) current-frame projection, pos (B, N, 2) -> (B, T, X, Y)."""
        B, P, X, Y = fs.shape
        T = self.offs.shape[0]
        q = (pos.unsqueeze(1) + self.offs.to(pos.dtype).view(1, T, 1, 2)).reshape(B, T * X * Y, 2)
        look = sample_at(gc, q, self.grid).view(B, P, T, X * Y)
        return (look * fs.flatten(2).unsqueeze(2)).sum(1).view(B, T, X, Y) / math.sqrt(P)

    def forward(self, src, cur, steps, v0, iters=4, correct=None):
        """src (B, C, X, Y) source map, cur (B, C, X, Y) current frame, v0 (B, 2, X, Y) m/s -> list of the velocity
        fields after each iteration (B, 2, X, Y). correct: optional function v -> v applied after EVERY iteration, e.g.
        a Doppler measurement update (doppler_update) - the refinement proposes, the measurement corrects (without it
        the refinement undid what Doppler fixed: stripes 0.07 -> 2.33 m/s over 4 iterations, 2026-10-26)."""
        B = src.shape[0]
        S = steps_tensor(steps, B, src.device, src.dtype)
        dt = S[:, :, 0].sum(1)
        dt = torch.where(dt.abs() < 1e-3, torch.full_like(dt, 1e-3), dt).view(B, 1, 1)
        th = (S[:, :, 0] * S[:, :, 2]).sum(1).view(B, 1, 1)
        c, s = torch.cos(th), torch.sin(th)
        fs, gc = self.f(src), self.g(cur)
        v, out = v0, []
        for _ in range(iters):
            v = v.detach()
            corr = self.correlation(fs, gc, landing(S, v, self.grid))
            d = self.net(torch.cat([corr, fs, v / self.v_scale], 1))               # metres, target axes
            d = self.step_max * torch.tanh(d / self.step_max)                      # inside the lookup window
            v = v + torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1) / dt.unsqueeze(1)
            if correct is not None:
                v = correct(v)
            out.append(v)
        return out


class FlowPyramid(nn.Module):
    """Coarse to fine (PWC-Net style): FlowRefine on pooled maps first - at 4x the cell size a +-3 cell window sees
    +-6 m instead of +-1.5 m, so large motions are found before the fine levels refine them. Per level: features
    average-pooled by the factor, a grid with factor x the cell size, its own update operator, `iters` iterations;
    the velocity (m/s, not cells) is pooled into the level, and the level's CORRECTION is upsampled (bilinear) and
    added to the fine velocity (the fine detail of v0 is kept; identity at initialisation). forward returns the
    fine-grid velocity after each iteration (sum(iters) entries). v is detached between iterations, also between
    levels (RAFT): each level learns only from the losses on its own outputs - put a loss on every output."""

    def __init__(self, cin, grid, factors=(4, 2, 1), iters=(2, 1, 1), radius=3, detach_levels=True, **kw):
        super().__init__()
        self.detach_levels = detach_levels    # False = the earlier behaviour (coarse corrections pass a gradient into
        #                                       v0); kept as an option: on the box test it scored better (one seed)
        assert len(factors) == len(iters)
        assert all(grid.nx % f == 0 and grid.ny % f == 0 for f in factors), "grid size must divide by every factor"
        self.grid, self.factors, self.iters = grid, tuple(factors), tuple(iters)
        self.grids = [Grid(x_min=grid.x_min, x_max=grid.x_max, y_min=grid.y_min, y_max=grid.y_max, res=grid.res * f)
                      for f in factors]
        self.levels = nn.ModuleList(FlowRefine(cin, g, radius=radius, **kw) for g in self.grids)

    def forward(self, src, cur, steps, v0, correct=None):
        """correct: as in FlowRefine, applied to every fine-grid output (inside a coarse level the next iteration does
        not see the correction of the previous one: it works on the pooled field)."""
        out, v = [], v0
        for f, n, rf in zip(self.factors, self.iters, self.levels):
            v = v.detach() if self.detach_levels else v   # between levels as between iterations (RAFT); without it the coarse output
            #                             base + up(vk - pool(base)) passed a high-pass gradient into v0 (the
            #                             hypotheses), and B collapsed in both pyramid arms (10 / 22 m/s, 2026-10-26)
            if f == 1:
                ks = rf(src, cur, steps, v, iters=n, correct=correct)
            else:                       # the coarse level adds its CORRECTION, upsampled: the fine detail of v stays
                vl, base = F.avg_pool2d(v, f), v
                ks = [base + F.interpolate(vk - vl, scale_factor=f, mode="bilinear", align_corners=False)
                      for vk in rf(F.avg_pool2d(src, f), F.avg_pool2d(cur, f), steps, vl, iters=n)]
                if correct is not None:
                    ks = [correct(k) for k in ks]
            out += ks
            v = ks[-1]
        return out


class GapFill(nn.Module):
    """Close the unknown cells after a splat: out = feat + (1 - min(mass, 1)) * g([feat, min(mass, 1)]), g a small
    dilated conv net (receptive field 31 cells with the default dilations). The last layer starts at zero, so the
    block starts as the identity, and a cell with mass >= 1 is never changed - the filler cannot hide a wrong motion
    in known cells, it only paints where nothing landed (new field of view, cells an object has left)."""

    def __init__(self, cin, hidden=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, c = [], cin + 1
        for d in dilations:
            layers += [nn.Conv2d(c, hidden, 3, padding=d, dilation=d), nn.ReLU()]
            c = hidden
        layers.append(nn.Conv2d(hidden, cin, 1))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, feat, mass):
        known = mass.clamp(max=1.0)
        return feat + (1 - known) * self.net(torch.cat([feat, known], 1))


def _selftest():
    torch.manual_seed(0)
    dt64 = torch.float64
    g = Grid(x_min=0, x_max=20, y_min=-10, y_max=10, res=0.5)                 # 40 x 40 cells
    tp = TemporalPropagation(cin=3, grid=g, n_hyp=2).double()                # bilinear kernel (default)
    x = torch.zeros(1, 3, g.nx, g.ny, dtype=dt64)
    x[0, :, 20, 20] = torch.tensor([1.0, 2.0, 3.0])
    # identity: no time step and static -> the map is unchanged (mass 1 where a source sits)
    tp.set_velocity((0.0, 0.0), static_weight=0.0)
    r = tp(x, [(0.0, 0.0, 0.0)])
    assert torch.allclose(r["feat"][0, :, 20, 20], x[0, :, 20, 20], atol=1e-9)
    # mass conservation: every source inside the grid contributes exactly 1
    assert abs(r["mass"].sum().item() - g.nx * g.ny) < 1e-6
    # Gaussian kernel: also mass conserving away from the border, and it keeps the centroid
    tg = TemporalPropagation(cin=3, grid=g, n_hyp=2, sigma0=1.5, radius=3).double()
    tg.set_velocity((0.0, 0.0), static_weight=0.0)
    rg = tg(x, [(0.0, 0.0, 0.0)])
    mg = rg["mass"][0, 0, 5:-5, 5:-5]
    assert torch.allclose(mg, torch.ones_like(mg), atol=1e-6)
    # a known velocity: 4 m/s forward for 0.5 s, no ego motion -> the spike moves 2 m = 4 cells forward
    tp.set_velocity((4.0, 0.0))
    r = tp(x, [(0.5, 0.0, 0.0)])
    assert r["feat"][0, 0].argmax().item() == 24 * g.ny + 20
    # ego motion 1 m forward, static world -> 2 cells back
    tp.set_velocity((0.0, 0.0), static_weight=0.0)
    r = tp(x, [(0.1, 10.0, 0.0)])
    assert r["feat"][0, 0].argmax().item() == 18 * g.ny + 20
    # aliasing: wrap(v.u - d) is 0 when d differs from v.u by 2 v_alias
    assert abs(wrap(torch.tensor(17.0) - torch.tensor(-3.0), 10.0).item()) < 1e-6
    # gradcheck through positions (velocity) and features on a tiny grid
    gs = Grid(x_min=0, x_max=4, y_min=-2, y_max=2, res=0.5)
    feat = torch.randn(1, 2, 5, dtype=dt64, requires_grad=True)
    pos = (torch.rand(1, 5, 2, dtype=dt64) * torch.tensor([2.0, 2.0], dtype=dt64) + torch.tensor([1.0, -1.0], dtype=dt64)).requires_grad_(True)
    wt = torch.rand(1, 5, dtype=dt64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: gaussian_splat(a, b, c, gs, 0.7, 2)[0], (feat, pos, wt), eps=1e-6, atol=1e-5)
    print("temporal selftest ok: identity, mass conservation, known velocity, ego motion, aliasing wrap, gradcheck "
          "(features, positions, hypothesis weights); full gates: tests/test_temporal.py")


if __name__ == "__main__":
    _selftest()

"""Motion block (t12, Mirko 2026-11-01): the velocity field at the current time k from TWO camera BEVs + ego motion + the
radar Doppler, in a small learned pyramid - the fused flow model of tests/test_flow_fused.py (plan 7.4, the arm without
correlation), moved from ground-truth maps onto the model's own BEV features.

    cur  = camera BEV of frame k          old = camera BEV of an older window frame, splatted to k by the EGO motion only
    E    = 1x1 projection + GroupNorm + ReLU + 3x3 conv (shared by cur and old): no own image encoder
    R    = fixed radar raster of k (no parameters): valid, log(1 + count), mean Doppler vr (linear, m/s), mean ray direction
           of the detections (unit), mean age
    per level l = 2 m, 1 m, 0.5 m (coarse to fine), v_up = the coarser level's velocity (0 at the start):
        W   = E(old) sampled where each cell of k was dt earlier under v_up (backward warp p - v_up dt)
        D   = Doppler residual u . v_up - vr at the cells with a detection
        v_l = v_up + net_l([E(cur), W, mass of old, v_up, R, D, ray direction / range of the cell])   (last layer zero-init)
Returns the velocity of every level at the grid resolution (coarse to fine); the last one is the output, the others take the
same loss (deep supervision). v in m/s, k's axes (x forward, y left), the convention of training/splat.py VelocityHead.
Fully differentiable: bilinear sampling and convolutions, no argmax, no matching.

Where it plugs in: t12 (frozen base): `training/train_motion_block.py` replaces the velocity head of the time mover
(TimeMover.motion) of a trained model by this block's output; the base stays frozen. t15 (joint): `ModelConfig.motion_block`
builds it inside the chain (`block_inputs` from the model's own camera frames, steps and radar; the field goes to the time
mover as `v_field`, with `box_vel_from_field` also into the object head's `vel`).
Run the self-test: python training/motion_block.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lift_pd import Grid
    from .splat import splat_window
except ImportError:                                   # run as a script
    from lift_pd import Grid
    from splat import splat_window

NR = 6                                                # radar raster channels


def radar_raster(det, B, grid):
    """detections {b, x, y, psi, vr, dt, ...} (vehicle frame at the output time; psi = ray direction from the sensor, vr =
    ego-compensated radial speed) -> (B, 6, X, Y): valid, log(1 + count), mean vr, mean ray direction (unit, 2), mean age.
    A fixed encoding: the cell a detection falls in, means over its detections; cells without a detection are 0."""
    X, Y = grid.nx, grid.ny
    dev = det["x"].device
    ix = torch.floor((det["x"] - grid.x_min) / grid.res).long()
    iy = torch.floor((det["y"] - grid.y_min) / grid.res).long()
    ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
    idx = (det["b"].long()[ok] * X + ix[ok]) * Y + iy[ok]
    N = B * X * Y

    def acc(w):
        return torch.zeros(N, device=dev, dtype=torch.float32).index_add_(0, idx, w.float()).view(B, X, Y)
    cnt = acc(torch.ones_like(idx, dtype=torch.float32))
    m = cnt.clamp(min=1.0)
    psi = det["psi"][ok]
    ux, uy = acc(torch.cos(psi)), acc(torch.sin(psi))
    un = torch.sqrt(ux ** 2 + uy ** 2).clamp(min=1e-6)
    has = (cnt > 0).float()
    return torch.stack([has, torch.log1p(cnt), acc(det["vr"][ok]) / m, ux / un * has, uy / un * has,
                        acc(det["dt"][ok]) / m], 1)


def pool_radar(r, s):
    """radar raster pooled to stride s: valid fraction, count, mean vr / ray / age over the detections in the coarse cell."""
    if s == 1:
        return r
    v = r[:, :1]
    pv = F.avg_pool2d(v, s)
    m = pv.clamp(min=1e-6)
    u = F.avg_pool2d(r[:, 3:5] * v, s)
    return torch.cat([pv, F.avg_pool2d(r[:, 1:2], s), F.avg_pool2d(r[:, 2:3] * v, s) / m,
                      u / u.norm(dim=1, keepdim=True).clamp(min=1e-6) * (pv > 0), F.avg_pool2d(r[:, 5:6] * v, s) / m], 1)


def cell_geometry(grid, s, device):
    """(3, X/s, Y/s): unit ray direction from the vehicle origin to each cell centre and range / 60 (a convolution is
    translation-equivariant and cannot know where a radar ray points)."""
    xs, ys = grid.xy_centres()
    x = xs.view(-1, s).mean(1) if s > 1 else xs
    y = ys.view(-1, s).mean(1) if s > 1 else ys
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
    return torch.stack([gx / r, gy / r, r / 60.0]).to(device)


def backward_warp(f, v, dt, res):
    """f (B, c, X, Y) in k's frame -> f sampled at p - v dt for every cell p (bilinear, zeros outside); v (B, 2, X, Y) m/s,
    dt (B,) s > 0 (the age of the older frame), res the cell size of f in m."""
    B, _, X, Y = f.shape
    ix = torch.arange(X, device=f.device, dtype=f.dtype).view(1, X, 1)
    iy = torch.arange(Y, device=f.device, dtype=f.dtype).view(1, 1, Y)
    d = dt.to(f).view(B, 1, 1)
    qx = ix - v[:, 0] * d / res                                         # source position in cells
    qy = iy - v[:, 1] * d / res
    grid = torch.stack([(qy + 0.5) / Y * 2 - 1, (qx + 0.5) / X * 2 - 1], -1)
    return F.grid_sample(f, grid, align_corners=False, padding_mode="zeros")


def block_inputs(frames, steps, radar, grid, old=-1):
    """the block's inputs from the model's own tensors: frames (B, F, C, X, Y) camera BEVs per window frame (0 = current, F >= 2),
    steps[k] the ego steps of frame k to the current time (a list of (dt, v, omega) tensors, as model.forward builds them),
    radar the batch's detections; old: the window frame used as the older frame (-1 = the oldest).
    -> (cur, the older frame ego-aligned to k (one splat, differentiable), its mass clamped to 1, radar raster, dt (B,) > 0)"""
    B, Fn = frames.shape[:2]
    k = old % Fn
    if k == 0:
        raise ValueError(f"the motion block needs an older camera frame (n_cam >= 2), got {Fn} frame(s)")
    if not steps[k]:
        raise ValueError(f"the older frame {k} has no ego steps to the current time")
    w, m = splat_window([(frames[:, k], steps[k])], grid, head=None, mass_gate=True)
    with torch.no_grad():
        rad = radar_raster(radar, B, grid)
    dt = sum(s[0] for s in steps[k]).abs().reshape(B)
    return frames[:, 0], w[:, 0], m[:, 0].clamp(max=1.0), rad, dt


def _block(cin, cout, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil), nn.GroupNorm(8, cout), nn.ReLU(True))


class Level(nn.Module):
    """one pyramid level: backward warp + Doppler comparison -> a learned velocity update (zero start)."""

    def __init__(self, s, c, res):
        super().__init__()
        self.s, self.res = s, res * s
        cin = 2 * c + 1 + 2 + NR + 1 + 3
        dil = (1, 2, 4) if s >= 4 else (1, 2)
        self.net = nn.Sequential(_block(cin, 64), *[_block(64, 64, d) for d in dil[1:]], nn.Conv2d(64, 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ek, eo, mass, v_up, rad, dt, geo):
        s = self.s
        fk, fo, m = (F.avg_pool2d(t, s) if s > 1 else t for t in (ek, eo, mass))
        w = backward_warp(fo, v_up, dt, self.res)
        r = pool_radar(rad, s)
        valid, vr, u = (r[:, :1] > 0).float(), r[:, 2:3], r[:, 3:5]
        resid = valid * ((u * v_up).sum(1, keepdim=True) - vr) / 10.0
        g = geo.expand(fk.shape[0], -1, -1, -1)
        o = 10.0 * self.net(torch.cat([fk, w, m, v_up / 10.0, r, resid, g], 1))
        return v_up + o


class MotionBlock(nn.Module):
    def __init__(self, c_in, grid, c=32, strides=(4, 2, 1)):
        super().__init__()
        self.grid, self.strides = grid, tuple(strides)
        assert all(grid.nx % s == 0 and grid.ny % s == 0 for s in self.strides), (grid.nx, grid.ny, strides)
        self.enc = nn.Sequential(nn.Conv2d(c_in, c, 1), nn.GroupNorm(8, c), nn.ReLU(True), _block(c, c))
        self.levels = nn.ModuleList(Level(s, c, grid.res) for s in self.strides)
        self._geo = {}

    def geo(self, s, device):
        if (s, device) not in self._geo:
            self._geo[(s, device)] = cell_geometry(self.grid, s, device)[None]
        return self._geo[(s, device)]

    def forward(self, cur, old, mass, rad, dt):
        """cur (B, C, X, Y) camera BEV of k; old (B, C, X, Y) an older frame ego-aligned to k, mass (B, 1, X, Y) where it
        landed; rad (B, 6, X, Y) radar_raster of k; dt (B,) age of the older frame in s (> 0).
        -> list of v (B, 2, X, Y) per level at the grid resolution, coarse to fine."""
        ek, eo = self.enc(cur), self.enc(old)
        B = cur.shape[0]
        s0 = self.strides[0]
        v = cur.new_zeros(B, 2, self.grid.nx // s0, self.grid.ny // s0)
        outs = []
        for i, lv in enumerate(self.levels):
            if i > 0:
                v = F.interpolate(v, scale_factor=self.strides[i - 1] // lv.s, mode="bilinear", align_corners=False)
            v = lv(ek, eo, mass, v, rad, dt, self.geo(lv.s, cur.device))
            outs.append(v if lv.s == 1 else F.interpolate(v, scale_factor=lv.s, mode="bilinear", align_corners=False))
        return outs


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    g = Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.5)          # the model grid, 120 x 104
    X, Y = g.nx, g.ny
    xs, ys = g.xy_centres()
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")

    def blob(cx, cy):
        return torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / 2.0)[None, None]
    # 1. warp sign: a blob at (30, 0) now that moved with v = (8, -4) m/s over dt = 0.5 s was at (26, 2): the old frame
    #    warped with the true v matches the current one, with v = 0 it does not
    cur, old = blob(30.0, 0.0), blob(26.0, 2.0)
    v = torch.tensor([8.0, -4.0]).view(1, 2, 1, 1).expand(1, 2, X, Y)
    dt = torch.tensor([0.5])
    e_true = (backward_warp(old, v, dt, g.res) - cur).abs().sum()
    e_zero = (backward_warp(old, torch.zeros_like(v), dt, g.res) - cur).abs().sum()
    assert e_true < 0.05 * e_zero, (float(e_true), float(e_zero))
    print(f"warp sign ok: |warp(old, v_true) - cur| = {float(e_true):.3f} vs v = 0: {float(e_zero):.3f}")
    # 2. radar raster: two detections in one cell average, one elsewhere; Doppler residual 0 for the radial truth
    det = {"b": torch.tensor([0, 0, 1]), "x": torch.tensor([10.1, 10.3, 40.0]), "y": torch.tensor([0.1, 0.2, -5.0]),
           "psi": torch.tensor([0.0, 0.0, -0.1244]), "vr": torch.tensor([2.0, 4.0, 5.0]), "dt": torch.tensor([0.1, 0.3, 0.0])}
    r = radar_raster(det, 2, g)
    i, j = int((10.1 - g.x_min) / g.res), int((0.1 - g.y_min) / g.res)
    assert r[0, 0, i, j] == 1 and abs(float(r[0, 2, i, j]) - 3.0) < 1e-6 and abs(float(r[0, 5, i, j]) - 0.2) < 1e-6
    assert abs(float(r[0, 3, i, j]) - 1.0) < 1e-6 and int(r[1, 0].sum()) == 1 and int(r[0, 0].sum()) == 1
    pr = pool_radar(r, 4)
    assert pr.shape == (2, NR, X // 4, Y // 4) and abs(float(pr[0, 2].max()) - 3.0) < 1e-5
    print("radar raster ok: mean Doppler, ray and age per cell, pooled per level")
    # 3. the block: zero start (v = 0 on every level), shapes, gradient reaches every learned part
    C = 64
    blk = MotionBlock(C, g)
    curf, oldf = torch.randn(2, C, X, Y), torch.randn(2, C, X, Y)
    out = blk(curf, oldf, torch.ones(2, 1, X, Y), r, torch.tensor([0.33, 0.33]))
    assert len(out) == 3 and all(o.shape == (2, 2, X, Y) for o in out) and all(float(o.detach().abs().max()) == 0 for o in out)
    sum(o.square().mean() + o.mean() for o in out).backward()
    # at the zero start only the last layer of each level gets a gradient (the rest is behind a zero layer); after one step all
    assert all(float(lv.net[-1].weight.grad.abs().sum()) > 0 for lv in blk.levels)
    opt = torch.optim.SGD(blk.parameters(), lr=0.1)
    opt.step()
    blk.zero_grad()
    out = blk(curf, oldf, torch.ones(2, 1, X, Y), r, torch.tensor([0.33, 0.33]))
    sum(o.square().mean() + o.mean() for o in out).backward()
    dead = [n for n, p in blk.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead
    print(f"block ok: 3 levels at {X} x {Y}, zero start, gradient reaches all {sum(1 for _ in blk.parameters())} tensors "
          f"after one step; {sum(p.numel() for p in blk.parameters()) / 1e3:.0f} k parameters")
    # 4. it can learn: fit the velocity of a moving blob from (cur, old) in a few hundred steps of Adam
    torch.manual_seed(1)
    blk = MotionBlock(2, g, c=16)
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    vt = torch.tensor([6.0, 3.0])
    for it in range(150):
        cx, cy = 15 + 30 * torch.rand(1).item(), -15 + 30 * torch.rand(1).item()
        c1, o1 = blob(cx, cy), blob(cx - vt[0].item() * 0.5, cy - vt[1].item() * 0.5)
        f_cur, f_old = torch.cat([c1, c1], 1), torch.cat([o1, o1], 1)
        m = c1[:, 0] > 0.3
        outs = blk(f_cur, f_old, torch.ones(1, 1, X, Y), torch.zeros(1, NR, X, Y), torch.tensor([0.5]))
        loss = sum((o.permute(0, 2, 3, 1)[m] - vt).abs().mean() for o in outs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    err = float((outs[-1].permute(0, 2, 3, 1)[m] - vt).norm(dim=-1).mean())
    assert err < 2.0, err
    print(f"learning ok: blob velocity error after 150 steps {err:.2f} m/s (|v| {float(vt.norm()):.1f})")
    # 5. block_inputs (the joint form, t15): the older frame is the window frame ego-aligned to k, dt its age, radar rastered;
    #    a pure forward drive of 10 m/s for 0.33 s moves a feature 3.3 m back; the oldest frame by default; one frame refused
    fr = torch.zeros(1, 3, 2, X, Y)
    fr[0, 2, 0, 60, 52] = 1.0                                             # a point at x = 30.25 m in the oldest frame
    steps = [[], [tuple(torch.tensor([t]) for t in (0.17, 10.0, 0.0))], [tuple(torch.tensor([t]) for t in (0.33, 10.0, 0.0))]]
    det = {"b": torch.zeros(1), "x": torch.tensor([20.1]), "y": torch.tensor([0.1]), "psi": torch.zeros(1), "vr": torch.tensor([-2.0]), "dt": torch.tensor([0.1])}
    cur, old_al, mass, rad, dt = block_inputs(fr, steps, det, g)
    assert cur.shape == (1, 2, X, Y) and abs(float(dt[0]) - 0.33) < 1e-6 and float(rad[0, 2, 40, 52]) == -2.0
    xc = float((old_al[0, 0] * gx).sum() / old_al[0, 0].sum())
    assert abs(xc - (30.25 - 3.3)) < 0.3, xc                               # the ego drove 3.3 m towards it
    for bad in ((fr[:, :1], steps[:1]), (fr, [[], [], []])):              # one frame; an older frame without ego steps
        try:
            block_inputs(bad[0], bad[1], det, g)
            raise AssertionError("must be refused")
        except ValueError:
            pass
    # turning and two hops: a point of the oldest frame lands where chain_positions puts it (the ego turns 0.4 rad/s, 10 m/s)
    from splat import chain_positions
    fr2 = torch.zeros(1, 3, 2, X, Y)
    fr2[0, 2, 0, 70, 60] = 1.0                                           # x 35.25 m, y 4.25 m
    hop = tuple(torch.tensor([t]) for t in (0.165, 10.0, 0.4))
    st2 = [[], [hop], [hop, hop]]
    _, al2, _, _, dt2 = block_inputs(fr2, st2, det, g)
    p_exp = chain_positions(torch.tensor([[[35.25, 4.25]]]), [hop, hop])[0, 0]
    xc2 = float((al2[0, 0] * gx).sum() / al2[0, 0].sum())
    yc2 = float((al2[0, 0] * gy).sum() / al2[0, 0].sum())
    assert abs(float(dt2[0]) - 0.33) < 1e-6 and abs(xc2 - float(p_exp[0])) < 0.3 and abs(yc2 - float(p_exp[1])) < 0.3, (xc2, yc2, p_exp)
    print(f"block_inputs ok: older frame ego-aligned (x {xc:.2f} m, expected {30.25 - 3.3:.2f}), dt, radar raster, one frame refused")
    print("motion_block self-test passed")


if __name__ == "__main__":
    _selftest()

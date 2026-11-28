"""Motion block (t12, the owner 2026-11-01): the velocity field at the current time k from TWO camera BEVs + ego motion + the
radar Doppler, in a small learned pyramid - the fused flow model of the early studies (tag v1; the arm without
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
same loss (deep supervision). v in m/s, k's axes (x forward, y left), the convention of net/model/splat.py VelocityHead.
Fully differentiable: bilinear sampling and convolutions, no argmax, no matching.

Where it plugs in: t12 (frozen base): `training/train_motion_block.py` replaces the velocity head of the time mover
(TimeMover.motion) of a trained model by this block's output; the base stays frozen. t15 (joint): `ModelConfig.motion_block`
builds it inside the chain (`block_inputs` from the model's own camera frames, steps and radar; the field goes to the time
mover as `v_field`, with `box_vel_from_field` also into the object head's `vel`).
Tests: tests/test_motion_block.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_fusion.net.model.splat import splat_window

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
    N = B * X * Y
    idx = (det["b"].long()[ok] * X + ix[ok]) * Y + iy[ok]

    def acc(w):
        return torch.zeros(N, device=dev, dtype=torch.float32).index_add_(0, idx, w.float()).view(B, X, Y)
    psi, vr, age = det["psi"][ok], det["vr"][ok], det["dt"][ok]
    cnt = acc(torch.ones_like(idx, dtype=torch.float32))
    m = cnt.clamp(min=1.0)
    ux, uy = acc(torch.cos(psi)), acc(torch.sin(psi))
    un = torch.sqrt(ux ** 2 + uy ** 2).clamp(min=1e-6)
    has = (cnt > 0).float()
    return torch.stack([has, torch.log1p(cnt), acc(vr) / m, ux / un * has, uy / un * has,
                        acc(age) / m], 1)


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

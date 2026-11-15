"""Fallbacks and control arms (bev_fusion/plan.md sections 2.3, 6, 7.1): proven blocks of the literature as drop-in
replacements that keep the interfaces of the blocks they replace.

- EgoWarpGRU (temporal control arm A, replaces TemporalFusion; FIERY 2104.10490, wayveai/fiery, MIT, "Copyright (c) 2021
  Wayve Technologies Limited" - the idea only, nothing copied, literature/code/fiery_ego_warp.md): the previous state is
  PULLED into the current ego frame by the inverse of the ego step(s) (dt, v, omega) of splat.splat_positions (same
  constant-turn-rate geometry: rotation exact, translation d = (v dt, v omega dt^2 / 2)), by one grid_sample whose
  coordinates are computed in METRES on our grid (FIERY's normalised warp assumes a square grid centred on the ego; ours
  is x 0..60 m, y -26..26 m), with a validity mask (source outside the old grid -> invalid, hidden state 0 there); then
  [cur, obs_mass, known, ego-motion channels (tx, ty, theta, dt) of the pull] -> ConvGRU cell (Ballas et al. 1511.06432,
  written from the equations) -> out = the new hidden state. `past` frames (FIERY's cumulative warp) are each pulled
  from their own original with their accumulated steps and run through the cell before `cur`, oldest first.
- LiftS (the splat lift, replaces LiftPD; LSS 2008.05711 - NVIDIA research licence, re-implemented, never copied;
  BEVPoolV2 2211.17111 / DualBEV (Apache-2.0) index semantics, literature/code/dualbev.md): a table per calibration and
  feature size holds, for every feature pixel (u, v) and depth bin d with a voxel, the flat source index and the voxel
  index (P_cam = z_d K^-1 ida^-1 [u, v, 1], P_car = T^-1 P_cam, voxel = floor of the cell coordinates, invalid points
  dropped); run time: scatter-add (index_add) of alpha_d * c into the voxels, height kept as channels (sum pooling, no
  normalisation: the total mass of the valid points is conserved). Same forward signature and output as LiftPD.
- CMXFuser / CMXTrunk (the two-stream fusion pyramid fallback of plan 6, replaces ConcatFusion + BEVTrunk's encoder;
  GaussianCaR, santimontiel/gaussiancar, Apache-2.0: the stage structure of gaussiancar/modeling/components/fuser.py is
  adapted - per stage a rectify module and a fusion module, stride-2 conv + 2 / 4 / 4 residual blocks per stream between
  stages; the rectify / fusion modules derive from the CMX paper's code (licence unchecked) and the residual blocks from
  Ultralytics-style blocks in its conv.py, so both are RE-IMPLEMENTED from the description in
  literature/code/gaussiancar_fusion.md, nothing copied): grid 120 x 104 -> 60 x 52 -> 30 x 26 -> 15 x 13, four fused
  scales; CMXTrunk adds a small top-down decoder back to 0.5 m and the semantic head of bev_trunk.py.
  Deviation: the cross-attention is the efficient attention of Shen et al. (WACV 2021; keys softmaxed over the cells,
  queries over the channels, context k^T v per head) - bounded at 12,480 cells, where the softmax of an unnormalised
  k^T v saturates.

Conventions as training/splat.py and lift_pd.py: vehicle frame x forward, y left, z up; maps (B, C, X, Y), x-major.
Run the self-test: python training/fallbacks.py   (CPU is enough)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bev_trunk import SemanticHead
    from .fusion import drop_modality
    from .lift_pd import Grid, height_to_channels
    from .splat import grid_points
except ImportError:                                   # run as a script
    from bev_trunk import SemanticHead
    from fusion import drop_modality
    from lift_pd import Grid, height_to_channels
    from splat import grid_points


# ============================================================================================ 1 EgoWarpGRU (arm A)
def _steps(step):
    """(dt, v, omega) or a list of them (oldest first) -> list of steps; None -> []."""
    if step is None:
        return []
    return [tuple(s) for s in step] if isinstance(step[0], (tuple, list)) else [tuple(step)]


def _col(t, like):
    return torch.as_tensor(t, dtype=like.dtype, device=like.device).reshape(-1, 1)


def pull_positions(p, steps):
    """Inverse of splat.chain_positions without object motion: p (B, N, 2) positions in the TARGET (current) frame,
    steps [(dt, v, omega), ...] from the source frame to the target, oldest first -> the same static points in the source
    frame (B, N, 2). Each step f(q) = R(-omega dt)(q - d) is inverted exactly: q = R(omega dt) p + d."""
    q = p
    for dt, v, om in reversed(steps):
        dt, v, om = _col(dt, p), _col(v, p), _col(om, p)
        th = om * dt
        c, s = torch.cos(th), torch.sin(th)
        qx = c * q[..., 0] - s * q[..., 1] + v * dt
        qy = s * q[..., 0] + c * q[..., 1] + 0.5 * v * om * dt * dt
        q = torch.stack([qx, qy], -1)
    return q


def ego_pull(feat, grid, steps, mode="bilinear"):
    """Pull a map of the source time (B, C, X, Y) into the target ego frame: grid_sample at the source positions of the
    target cell centres, coordinates normalised per axis in metres (no square / centred grid assumed, align_corners=False).
    Returns (pulled (B, C, X, Y), valid (B, 1, X, Y): 1 where the source point lies inside the old grid, else 0 and the
    pulled value 0, ego (B, 4): (tx, ty, theta, dt) = the target origin in the source frame, total turn, total dt)."""
    B, C, X, Y = feat.shape
    steps = _steps(steps) if steps and not isinstance(steps, list) else (steps or [])
    pts = grid_points(grid, feat.device, feat.dtype).unsqueeze(0).expand(B, -1, -1)
    src = pull_positions(pts, steps)
    gx = 2 * (src[..., 0] - grid.x_min) / (grid.nx * grid.res) - 1                   # along X (dim 2 = H)
    gy = 2 * (src[..., 1] - grid.y_min) / (grid.ny * grid.res) - 1                   # along Y (dim 3 = W)
    valid = ((src[..., 0] >= grid.x_min) & (src[..., 0] <= grid.x_max) & (src[..., 1] >= grid.y_min)
             & (src[..., 1] <= grid.y_max)).to(feat.dtype).view(B, 1, X, Y)
    g = torch.stack([gy, gx], -1).view(B, X, Y, 2)
    out = F.grid_sample(feat, g, mode=mode, padding_mode="border", align_corners=False) * valid
    o = pull_positions(feat.new_zeros(B, 1, 2), steps)[:, 0]
    th = sum((_col(om, feat) * _col(dt, feat))[:, 0] for dt, _, om in steps) if steps else feat.new_zeros(B)
    T = sum(_col(dt, feat)[:, 0] for dt, _, _ in steps) if steps else feat.new_zeros(B)
    return out, valid, torch.stack([o[:, 0], o[:, 1], th.expand(B), T.expand(B)], 1)


class ConvGRUCell(nn.Module):
    """h' = (1 - z) h + z n, z, r = sigmoid(conv([x, h])), n = tanh(conv([x, r h])) (Ballas et al. 1511.06432)."""

    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.gates = nn.Conv2d(cin + ch, 2 * ch, k, 1, k // 2)
        self.cand = nn.Conv2d(cin + ch, ch, k, 1, k // 2)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class EgoWarpGRU(nn.Module):
    """Temporal control arm A: exact ego pull of the previous state + ConvGRU; interface of TemporalFusion.

    State: {"h": hidden (B, c_out, X, Y), "known": (B, 1, X, Y) where the hidden state has seen an observation}. The GRU
    input is [frame (c_in), obs_mass, known, ego (tx, ty, theta, dt broadcast)], so it can tell unknown from empty."""

    N_EXTRA = 6

    def __init__(self, c_in, c_out, grid, k=3):
        super().__init__()
        self.grid, self.c_in, self.c_out = grid, c_in, c_out
        self.cell = ConvGRUCell(c_in + self.N_EXTRA, c_out, k)

    def init_state(self, B, device=None, dtype=torch.float32):
        X, Y = self.grid.nx, self.grid.ny
        return {"h": torch.zeros(B, self.c_out, X, Y, device=device, dtype=dtype),
                "known": torch.zeros(B, 1, X, Y, device=device, dtype=dtype)}

    @staticmethod
    def detach_state(state):
        return {k: v.detach() for k, v in state.items()}

    def _update(self, x, obs, known, ego, h):
        B, _, X, Y = x.shape
        e = ego.view(B, 4, 1, 1).expand(B, 4, X, Y)
        return self.cell(torch.cat([x, obs, known, e], 1), h)

    def forward(self, cur, past=(), state=None, step=None, obs_mass=None):
        """cur (B, C, X, Y); past: list of (bev, steps) original past frames with their per-frame ego steps to the
        current time (as splat_window), run through the cell before cur, oldest first; state: the dict of the previous
        call or None; step: (dt, v, omega) or a list of them from the previous call's frame to this one; obs_mass
        (B, 1, X, Y) where the sensor observed (default 1). Returns out (B, c_out, X, Y), the new state and aux
        (h_warped, valid, known, ego)."""
        B, C, X, Y = cur.shape
        state = state or self.init_state(B, cur.device, cur.dtype)
        steps = _steps(step)
        hk, valid, ego = ego_pull(torch.cat([state["h"], state["known"]], 1), self.grid, steps)
        h, known = hk[:, :-1], hk[:, -1:].clamp(0, 1)
        h_warped = h
        for bev, st in past:
            xk, vk, ek = ego_pull(bev, self.grid, list(st))
            h = self._update(xk, vk, known, ek, h)
            known = torch.maximum(known, vk)
        om = obs_mass if obs_mass is not None else cur.new_ones(B, 1, X, Y)
        h = self._update(cur, om, known, ego, h)
        known = torch.maximum(known, om.clamp(0, 1))
        aux = {"h_warped": h_warped, "valid": valid, "known": known, "ego": ego}
        return h, {"h": h, "known": known}, aux


# ============================================================================================ 2 LiftS (splat lift)
def build_s_table(grid, K, T_cam_car, image_hw, feat_hw, depth_centres, ida=None, slice_height=0.5):
    """Splat table of one camera: for every feature pixel (u, v) and depth bin d the voxel its point falls into.

    Feature pixel j covers the augmented image as in lift_pd (centre at image coordinate (j + 0.5) W / Wf - 0.5); the
    point of bin d lies at depth z_d (bin centre) along the optical axis: P_cam = z_d K^-1 ida^-1 [u, v, 1],
    P_car = T_cam_car^-1 P_cam; voxel = (floor((x - x_min) / res), floor((y - y_min) / res), floor((z - z_lo) / dz))
    with the height slices centred on grid.z_levels (uniform spacing dz; slice_height if there is one level).
    Returns dict: src (N,) flat index into (D, Hf, Wf), pix (N,) flat index into (Hf, Wf), vox (N,) flat voxel index into
    (X, Y, Z) (x-major as lift_pd), shape (X, Y, Z), points (D*Hf*Wf, 3) all car points (for tests)."""
    dt = torch.float64
    dev = depth_centres.device
    H, W = image_hw
    Hf, Wf = feat_hw
    dc = depth_centres.to(dev, dt)
    D = len(dc)
    u = (torch.arange(Wf, device=dev, dtype=dt) + 0.5) * W / Wf - 0.5
    v = (torch.arange(Hf, device=dev, dtype=dt) + 0.5) * H / Hf - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1).reshape(-1, 3)                 # (Hf*Wf, 3)
    A = K.to(dev, dt) if ida is None else ida.to(dev, dt) @ K.to(dev, dt)
    ray = pix @ torch.linalg.inv(A).T                                                   # z component 1
    pc = dc.view(D, 1, 1) * ray.view(1, -1, 3)                                         # (D, Hf*Wf, 3)
    Tinv = torch.linalg.inv(T_cam_car.to(dev, dt))
    pts = (pc.reshape(-1, 3) @ Tinv[:3, :3].T + Tinv[:3, 3]).contiguous()                  # (D*Hf*Wf, 3)
    zl = torch.as_tensor(grid.z_levels, dtype=dt, device=dev)
    Z = len(zl)
    dz = float(zl[1] - zl[0]) if Z > 1 else slice_height
    assert Z == 1 or torch.allclose(zl[1:] - zl[:-1], torch.full_like(zl[1:], dz)), "uniform height slices"
    z_lo = float(zl[0]) - dz / 2
    X, Y = grid.nx, grid.ny
    ix = torch.floor((pts[:, 0] - grid.x_min) / grid.res).long()
    iy = torch.floor((pts[:, 1] - grid.y_min) / grid.res).long()
    iz = torch.floor((pts[:, 2] - z_lo) / dz).long()
    ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    src = torch.nonzero(ok)[:, 0]
    return {"src": src, "pix": src % (Hf * Wf), "vox": (ix[ok] * Y + iy[ok]) * Z + iz[ok], "shape": (X, Y, Z),
            "points": pts}


def lift_s(feat, depth_prob, table):
    """feat (B, C, Hf, Wf), depth_prob (B, D, Hf, Wf) -> volume (B, C, X, Y, Z): each valid (pixel, bin) adds
    p_d * c into its voxel (index_add: the BEVPoolV2 sum without a sorted interval list)."""
    B, C = feat.shape[:2]
    X, Y, Z = table["shape"]
    f = feat.reshape(B, C, -1)[:, :, table["pix"]]                                     # (B, C, N)
    p = depth_prob.reshape(B, -1)[:, table["src"]]                                     # (B, N)
    vol = feat.new_zeros(B, C, X * Y * Z).index_add(2, table["vox"], f * p.unsqueeze(1))
    return vol.view(B, C, X, Y, Z)


class LiftS(nn.Module):
    """Lift S for one camera; same forward signature and output (B, Z*C, X, Y) as LiftPD. The table is cached per
    calibration and feature size (no learned heights: a splat gives no gradient to the slice positions)."""

    def __init__(self, grid, depth_centres, slice_height=0.5):
        super().__init__()
        self.grid, self.slice_height = grid, slice_height
        self.register_buffer("depth_centres", torch.as_tensor(depth_centres, dtype=torch.float32))
        self._cache = None

    def table(self, feat_hw, K, T_cam_car, image_hw, ida=None):
        key = (tuple(K.flatten().tolist()), tuple(T_cam_car.flatten().tolist()), tuple(image_hw), tuple(feat_hw),
               None if ida is None else tuple(ida.flatten().tolist()), str(self.depth_centres.device))
        if self._cache is None or self._cache[0] != key:
            tab = build_s_table(self.grid, K, T_cam_car, image_hw, feat_hw, self.depth_centres, ida, self.slice_height)
            tab.pop("points")
            self._cache = (key, tab)
        return self._cache[1]

    def forward(self, feat, depth_prob, K, T_cam_car, image_hw, ida=None, key=None):
        """key: accepted for the signature of LiftPD (the callers pass the projection-table cache key when the batch has host copies); this lift keeps its own key
        (it holds the feature size too) and makes it as before"""
        assert depth_prob.shape[1] == len(self.depth_centres)
        tab = self.table(tuple(feat.shape[-2:]), K, T_cam_car, image_hw, ida)
        return height_to_channels(lift_s(feat, depth_prob, tab))


# ============================================================================================ 3 CMXFuser (GaussianCaR)
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.SiLU(True) if act else nn.Identity())


class ResBlock(nn.Module):
    """x + [conv-BN-SiLU, conv-BN, squeeze-excitation, dropout](x) (the block shape of GaussianCaR's fuser, re-written)."""

    def __init__(self, c, se_reduction=16, p_drop=0.1):
        super().__init__()
        h = max(c // se_reduction, 4)
        self.body = nn.Sequential(ConvBNAct(c, c), ConvBNAct(c, c, act=False))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1, bias=False), nn.ReLU(True),
                                nn.Conv2d(h, c, 1, bias=False), nn.Sigmoid())
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        y = self.body(x)
        return x + self.drop(y * self.se(y))


def _down(c, n_res):
    return nn.Sequential(ConvBNAct(c, c, 3, 2), *[ResBlock(c) for _ in range(n_res)])


class Rectify(nn.Module):
    """Cross-gating of the two streams: per-channel gates (avg + max pool of [c, r] -> MLP -> sigmoid) and per-cell gates
    (1x1 convs on [c, r] -> sigmoid), c' = c + lam_c g_c[1] r + lam_s g_s[1] r, r' = r + lam_c g_c[0] c + lam_s g_s[0] c.
    (The same operation as fusion.GateFusion.rectify without its input projections.)"""

    def __init__(self, c, reduction=1, lam_c=0.5, lam_s=0.5):
        super().__init__()
        h = max(c // reduction, 4)
        self.channel = nn.Sequential(nn.Linear(4 * c, h), nn.ReLU(True), nn.Linear(h, 2 * c), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2 * c, h, 1), nn.ReLU(True), nn.Conv2d(h, 2, 1), nn.Sigmoid())
        self.c, self.lam_c, self.lam_s = c, lam_c, lam_s

    def forward(self, cam, rad):
        cat = torch.cat([cam, rad], 1)
        cw = self.channel(torch.cat([cat.mean((2, 3)), cat.amax((2, 3))], 1)).view(-1, 2, self.c, 1, 1)
        sw = self.spatial(cat).unsqueeze(2)                                              # (B, 2, 1, X, Y)
        return (cam + self.lam_c * cw[:, 1] * rad + self.lam_s * sw[:, 1] * rad,
                rad + self.lam_c * cw[:, 0] * cam + self.lam_s * sw[:, 0] * cam)


class CrossAttention(nn.Module):
    """Efficient cross-attention (Shen et al., linear in the number of cells): per stream q, k, v; context of stream i
    ctx_i = softmax_cells(k_i)^T v_i (heads x d x d); stream 1 reads softmax_channels(q_1) ctx_2 and vice versa."""

    def __init__(self, c, heads=2):
        super().__init__()
        assert c % heads == 0
        self.heads = heads
        self.q1, self.q2 = nn.Linear(c, c, bias=False), nn.Linear(c, c, bias=False)
        self.kv1, self.kv2 = nn.Linear(c, 2 * c, bias=False), nn.Linear(c, 2 * c, bias=False)

    def _split(self, t):
        B, N, C = t.shape
        return t.view(B, N, self.heads, C // self.heads).transpose(1, 2)                  # (B, h, N, d)

    def _ctx(self, kv):
        k, v = (self._split(t) for t in kv.chunk(2, -1))
        return k.softmax(2).transpose(2, 3) @ v                                           # (B, h, d, d)

    def forward(self, u1, u2):
        B, N, C = u1.shape
        q1, q2 = self._split(self.q1(u1)).softmax(-1), self._split(self.q2(u2)).softmax(-1)
        c1, c2 = self._ctx(self.kv1(u1)), self._ctx(self.kv2(u2))
        merge = lambda t: t.transpose(1, 2).reshape(B, N, C)
        return merge(q1 @ c2), merge(q2 @ c1)


class FusionModule(nn.Module):
    """Two streams -> one map: per stream Linear to 2C (ReLU) split into y and u; v = cross-attention(u1, u2);
    x_i' = LN(x_i + Linear([y_i, v_i])); channel embed of [x_1', x_2']: BN(1x1(2C -> C) + 1x1 - depthwise 3x3 - ReLU - 1x1)."""

    def __init__(self, c, heads=2):
        super().__init__()
        self.proj1, self.proj2 = nn.Linear(c, 2 * c), nn.Linear(c, 2 * c)
        self.attn = CrossAttention(c, heads)
        self.end1, self.end2 = nn.Linear(2 * c, c), nn.Linear(2 * c, c)
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.res = nn.Conv2d(2 * c, c, 1, bias=False)
        self.embed = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True),
                                   nn.ReLU(True), nn.Conv2d(c, c, 1, bias=False))
        self.bn = nn.BatchNorm2d(c)

    def forward(self, cam, rad):
        B, C, X, Y = cam.shape
        x1, x2 = cam.flatten(2).transpose(1, 2), rad.flatten(2).transpose(1, 2)            # (B, N, C)
        y1, u1 = F.relu(self.proj1(x1)).chunk(2, -1)
        y2, u2 = F.relu(self.proj2(x2)).chunk(2, -1)
        v1, v2 = self.attn(u1, u2)
        x1 = self.ln1(x1 + self.end1(torch.cat([y1, v1], -1)))
        x2 = self.ln2(x2 + self.end2(torch.cat([y2, v2], -1)))
        m = torch.cat([x1, x2], -1).transpose(1, 2).reshape(B, 2 * C, X, Y)
        return self.bn(self.res(m) + self.embed(m))


class CMXFuser(nn.Module):
    """GaussianCaR's four-stage camera-radar fusion pyramid (stage structure adapted from its fuser.py, Apache-2.0).

    cam (B, c_cam, X, Y), rad (B, c_rad, X, Y) -> 1x1 projections to `embed` channels (no bias: a zero radar input is a
    zero radar stream) -> per stage s: (c, r) = Rectify(c, r), fused_s = FusionModule(c, r); between stages each stream
    is down-sampled by a stride-2 conv + 2 / 4 / 4 residual blocks. Returns the four fused scales [(B, embed, X / 2^s,
    Y / 2^s) for s = 0..3] (120 x 104 -> 60 x 52 -> 30 x 26 -> 15 x 13 on the default grid)."""

    N_RES = (2, 4, 4)

    def __init__(self, c_cam, c_rad, embed=128, heads=2, reduction=1):
        super().__init__()
        self.pc = nn.Conv2d(c_cam, embed, 1, bias=False)
        self.pr = nn.Conv2d(c_rad, embed, 1, bias=False)
        self.rect = nn.ModuleList([Rectify(embed, reduction) for _ in range(4)])
        self.fuse = nn.ModuleList([FusionModule(embed, heads) for _ in range(4)])
        self.cam_down = nn.ModuleList([_down(embed, n) for n in self.N_RES])
        self.rad_down = nn.ModuleList([_down(embed, n) for n in self.N_RES])
        self.embed = embed

    def forward(self, cam, rad):
        c, r = self.pc(cam), self.pr(rad)
        out = []
        for s in range(4):
            if s:
                c, r = self.cam_down[s - 1](c), self.rad_down[s - 1](r)
            c, r = self.rect[s](c, r)
            out.append(self.fuse[s](c, r))
        return out


class CMXDecoder(nn.Module):
    """Small top-down decoder over the four fused scales: lateral 1x1 convs, bilinear up-sampling to the next finer scale,
    sum, 3x3 conv-BN-ReLU; output (B, c_out, X, Y) at the finest scale (0.5 m)."""

    def __init__(self, embed, c_out):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(embed, c_out, 1, bias=False) for _ in range(4)])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
                                                   nn.ReLU(True)) for _ in range(3)])
        self.out_channels = c_out

    def forward(self, feats):
        y = self.lat[3](feats[3])
        for s in (2, 1, 0):
            y = F.interpolate(y, size=feats[s].shape[-2:], mode="bilinear", align_corners=False) + self.lat[s](feats[s])
            y = self.smooth[s](y)
        return y


class CMXTrunk(nn.Module):
    """Replaces ConcatFusion + BEVTrunk: features(cam, rad) -> (B, c_out, X, Y) at 0.5 m (as BEVTrunk.features),
    forward(cam, rad) -> class logits (B, n_classes, X, Y); modality dropout in training as ConcatFusion."""

    def __init__(self, c_cam, c_rad, n_classes=14, embed=128, c_out=128, heads=2, p_drop=0.2, head_hidden=None):
        super().__init__()
        self.fuser = CMXFuser(c_cam, c_rad, embed, heads)
        self.decoder = CMXDecoder(embed, c_out)
        self.head = SemanticHead(c_out, n_classes, head_hidden)
        self.p_drop = p_drop

    def pyramid(self, cam, rad):
        if self.training and self.p_drop > 0:
            cam, rad = drop_modality(cam, rad, self.p_drop)
        return self.fuser(cam, rad)

    def features(self, cam, rad):
        return self.decoder(self.pyramid(cam, rad))

    def forward(self, cam, rad):
        return self.head(self.features(cam, rad))


# ----------------------------------------------------------------------------------------------- self-test
def _test_ego_warp_gru():
    try:
        from .splat import splat_positions
        from .temporal_fusion import TemporalFusion
    except ImportError:
        from splat import splat_positions
        from temporal_fusion import TemporalFusion
    dt64 = torch.float64
    grid = Grid()
    X, Y = grid.nx, grid.ny
    pts = grid_points(grid, dtype=dt64).unsqueeze(0)

    # 1 one metre of forward ego motion moves a static pattern back by exactly 2 cells; the last 2 rows are invalid
    m = torch.randn(1, 3, X, Y, dtype=dt64)
    out, valid, ego = ego_pull(m, grid, [(0.1, 10.0, 0.0)])
    assert torch.allclose(out[:, :, :-2], m[:, :, 2:], atol=1e-12), "1 m forward = 2 cells"
    assert (valid[0, 0, -2:] == 0).all() and (valid[0, 0, :-2] == 1).all() and (out[:, :, -2:] == 0).all()
    assert torch.allclose(ego[0], torch.tensor([1.0, 0.0, 0.0, 0.1], dtype=dt64))
    # 2 a pure rotation lands points where splat_positions says: the pull positions invert splat_positions exactly, and
    #   a linear world (bilinear sampling is exact for it) is sampled there; a blob's centroid lands at the formula
    rot = [(0.5, 0.0, 0.4)]
    src = pull_positions(pts, rot)
    assert torch.allclose(splat_positions(src, 0.5, 0.0, 0.4), pts, atol=1e-12), "pull = inverse of splat_positions"
    lin = lambda p: 0.3 * p[..., 0] - 0.7 * p[..., 1] + 2.0
    out, valid, _ = ego_pull(lin(pts).view(1, 1, X, Y), grid, rot)
    xs, ys = grid.xy_centres(dtype=dt64)
    inner = ((src[..., 0] >= xs[0]) & (src[..., 0] <= xs[-1]) & (src[..., 1] >= ys[0]) & (src[..., 1] <= ys[-1])).view(X, Y)
    assert inner.sum() > 5000 and torch.allclose(out[0, 0][inner], lin(src).view(X, Y)[inner], atol=1e-9), "rotation"
    inside = ((src[..., 0] >= 0) & (src[..., 0] <= 60) & (src[..., 1] >= -26) & (src[..., 1] <= 26)).view(X, Y)
    assert torch.equal(valid[0, 0].bool(), inside) and (~inside).sum() > 100, "cells outside the old grid are invalid"
    assert (out[0, 0][~inside] == 0).all()
    p0 = torch.tensor([[[30.0, 8.0]]], dtype=dt64)
    blob = torch.exp(-((pts - p0) ** 2).sum(-1) / (2 * 1.5 ** 2)).view(1, 1, X, Y)
    bo, _, _ = ego_pull(blob, grid, rot)
    w = bo[0, 0]
    cx, cy = (w.sum(1) @ xs / w.sum()).item(), (w.sum(0) @ ys / w.sum()).item()
    ex = splat_positions(p0, 0.5, 0.0, 0.4)[0, 0]
    rot_err = math.hypot(cx - ex[0].item(), cy - ex[1].item())
    assert rot_err < 0.02, f"blob centroid after the rotation {rot_err:.4f} m"
    # 3 general steps (speed + turn, two steps) also invert the splat positions; the state is pulled, not resampled twice
    st = [(1 / 12, 12.0, 0.3), (1 / 12, 14.0, -0.2)]
    q = pull_positions(pts, st)
    fwd = splat_positions(splat_positions(q, *st[0]), *st[1])
    assert torch.allclose(fwd, pts, atol=1e-12), "two steps"

    # 4 the module: TemporalFusion's call pattern; a state pulled with the step, unknown cells marked, gradients
    torch.manual_seed(0)
    g = Grid(x_min=0.0, x_max=24.0, y_min=-8.0, y_max=8.0, res=0.5)
    Xs, Ys = g.nx, g.ny
    mod = EgoWarpGRU(2, 8, g).double()
    assert mod.cell.gates.in_channels == 2 + 6 + 8
    state, outs = None, []
    for t in range(3):
        step = (0.1, 10.0, 0.2) if t else None
        cur = torch.rand(2, 2, Xs, Ys, dtype=dt64)
        past = [(torch.rand(2, 2, Xs, Ys, dtype=dt64), [(0.1, 10.0, 0.2)])] if t == 2 else ()
        om = torch.ones(2, 1, Xs, Ys, dtype=dt64)
        om[:, :, :4] = 0                                               # the sensor does not see the first 2 m
        out, state, aux = mod(cur, past, state, step, om)
        outs.append(out)
        if t == 0:
            assert (aux["known"][:, :, :4] == 0).all() and (aux["known"][:, :, 4:] == 1).all(), "known = observed"
    assert out.shape == (2, 8, Xs, Ys) and torch.isfinite(out).all()
    assert aux["valid"][0, 0, -2:].sum() == 0, "cells the old grid did not cover are invalid"
    hw_ref, _, _ = ego_pull(torch.cat([mod.detach_state(state)["h"]], 1), g, [(0.1, 10.0, 0.2)])
    _, st2, aux2 = mod(torch.rand(2, 2, Xs, Ys, dtype=dt64), (), state, (0.1, 10.0, 0.2))
    assert torch.allclose(aux2["h_warped"], hw_ref, atol=1e-12), "the state is pulled by the step"
    assert (aux2["h_warped"][:, :, -2:] == 0).all(), "invalid cells hold no state"
    sum(o.square().mean() for o in outs).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), "gradients reach the GRU"
    tf = TemporalFusion(2, 8, g)
    o_tf, _, _ = tf(torch.rand(2, 2, Xs, Ys), (), None, None, torch.ones(2, 1, Xs, Ys))
    o_a, _, _ = EgoWarpGRU(2, 8, g)(torch.rand(2, 2, Xs, Ys), (), None, None, torch.ones(2, 1, Xs, Ys))
    assert o_tf.shape == o_a.shape, "drop-in for TemporalFusion"
    return rot_err


def _test_lift_s():
    try:
        from .lift_pd import LiftPD, _front_camera
    except ImportError:
        from lift_pd import LiftPD, _front_camera
    dt64 = torch.float64
    grid = Grid()
    dc = torch.arange(1.25, 60.0, 0.5, dtype=dt64)              # 118 bins, centres on voxel centres in x
    D = len(dc)
    K, T, hw = _front_camera(dt64)
    Hf, Wf = 14, 25                                              # stride 32: one feature pixel >= one cell from 10 m on
    X, Y, Z = grid.nx, grid.ny, len(grid.z_levels)

    # 1 geometry: the table's car point of pixel (u, v) at bin d is the analytic ray point of this camera
    tab = build_s_table(grid, K, T, hw, (Hf, Wf), dc)
    i, j, d = 7, 12, 60
    u, v = (j + 0.5) * 800 / Wf - 0.5, (i + 0.5) * 448 / Hf - 0.5
    z = dc[d].item()
    exp = torch.tensor([z, -(u - 399.5) / 633 * z, 1.5 - (v - 223.5) / 633 * z], dtype=dt64)
    assert torch.allclose(tab["points"][(d * Hf + i) * Wf + j], exp, atol=1e-9), "table geometry"

    # 2 one-hot depth, a feature that is 1 at one pixel: S writes 1 into ONE voxel, which is in the support of PD on the
    #   same calibration and depth bins, and PD's weighted centre is within one cell of it in x and y
    lpd, ls = LiftPD(grid, dc), LiftS(grid, dc)
    n_cmp = 0
    for (i, j, d) in [(6, 8, 21), (7, 12, 47), (6, 15, 77), (7, 18, 107), (6, 11, 60)]:
        feat = torch.zeros(1, 1, Hf, Wf, dtype=dt64)
        feat[0, 0, i, j] = 1
        one = torch.zeros(1, D, Hf, Wf, dtype=dt64)
        one[:, d] = 1
        vs = ls(feat, one, K, T, hw).view(1, Z, X, Y)[0]
        vp = lpd(feat, one, K, T, hw).view(1, Z, X, Y)[0]
        assert abs(vs.sum().item() - 1) < 1e-12 and (vs > 0).sum() == 1, "S: one voxel"
        k = torch.nonzero(vs > 0)[0]
        assert vp[tuple(k)] > 0, f"S voxel in PD support ({i}, {j}, {d})"
        idx = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt64) for n in (Z, X, Y)], indexing="ij"), 0)
        cen = (idx * vp).flatten(1).sum(1) / vp.sum()
        # in x, y; in z PD spreads over one feature pixel (2 m at 40 m for stride 32) and is cut at the top slice
        assert (cen - k.to(dt64))[1:].abs().max() <= 1.0, f"PD centre vs S voxel {cen.tolist()} {k.tolist()}"
        n_cmp += 1

    # 3 uniform / random depth: the total mass is conserved - on a grid that holds the whole frustum every pixel adds
    #   sum_c feat exactly; on the default grid the mass equals that of the valid (pixel, bin) points
    big = Grid(x_min=0.0, x_max=61.0, y_min=-40.0, y_max=40.0, res=1.0,
               z_levels=tuple(float(z) for z in torch.arange(-21.5, 24.0, 1.0)))
    tb = build_s_table(big, K, T, hw, (Hf, Wf), dc, slice_height=1.0)
    assert len(tb["src"]) == D * Hf * Wf, "the big grid holds every point"
    feat = torch.rand(2, 3, Hf, Wf, dtype=dt64)
    uni = torch.full((2, D, Hf, Wf), 1.0 / D, dtype=dt64)
    rnd = torch.softmax(torch.randn(2, D, Hf, Wf, dtype=dt64), 1)
    for p in (uni, rnd):
        vol = lift_s(feat, p, tb)
        assert torch.allclose(vol.sum((2, 3, 4)), feat.sum((2, 3)), atol=1e-9), "mass conservation"
    vol = lift_s(feat, uni, tab)
    ok = torch.zeros(D * Hf * Wf, dtype=dt64)
    ok[tab["src"]] = 1.0
    ref = (feat.view(2, 3, 1, -1) * (ok.view(D, -1) / D)).sum((2, 3))
    assert torch.allclose(vol.sum((2, 3, 4)), ref, atol=1e-9) and 0 < len(tab["src"]) < D * Hf * Wf

    # 4 rig independence: camera 1 m further forward -> BEV shifted by 2 cells
    K2, T2, _ = _front_camera(dt64, x0=1.0)
    a = ls(feat, rnd, K, T, hw)
    c = LiftS(grid, dc)(feat, rnd, K2, T2, hw)
    assert torch.allclose(c[:, :, 2:], a[:, :, :-2], atol=1e-12), "rig shift"

    # 5 output = LiftPD's interface; gradients reach the feature and the depth distribution (gradcheck, small grid)
    pd_out = lpd(feat, rnd, K, T, hw)
    assert a.shape == pd_out.shape == (2, Z * 3, X, Y)
    small = Grid(x_min=4, x_max=8, y_min=-1, y_max=1, res=1.0, z_levels=(0.0, 1.0))
    Ks = torch.tensor([[10.0, 0, 5.5], [0, 10.0, 3.0], [0, 0, 1]], dtype=dt64)
    dcs = torch.linspace(3.0, 9.0, 6, dtype=dt64)
    ts = build_s_table(small, Ks, T, (7, 12), (7, 12), dcs, slice_height=1.0)
    f_s = torch.randn(1, 2, 7, 12, dtype=dt64, requires_grad=True)
    p_s = torch.softmax(torch.randn(1, 6, 7, 12, dtype=dt64), 1).requires_grad_(True)
    assert len(ts["src"]) > 0 and torch.autograd.gradcheck(lambda f, p: lift_s(f, p, ts), (f_s, p_s), eps=1e-6, atol=1e-5)
    # float32 module on float32 inputs (the table is built in float64 and cached)
    o32 = LiftS(grid, dc.float())(feat.float(), rnd.float(), K.float(), T.float(), hw)
    assert o32.dtype == torch.float32 and torch.allclose(o32.double(), a, atol=1e-5)
    return n_cmp


def _test_cmx():
    torch.manual_seed(0)
    B, E = 2, 16
    m = CMXTrunk(12, 5, n_classes=7, embed=E, c_out=24, p_drop=0.0)
    with torch.no_grad():                                         # shapes on the default grid
        fused = m.pyramid(torch.randn(1, 12, 120, 104), torch.randn(1, 5, 120, 104))
        assert [tuple(f.shape) for f in fused] == [(1, E, 120, 104), (1, E, 60, 52), (1, E, 30, 26), (1, E, 15, 13)]
        assert m(torch.randn(1, 12, 120, 104), torch.randn(1, 5, 120, 104)).shape == (1, 7, 120, 104)
    cam, rad = torch.randn(B, 12, 48, 40), torch.randn(B, 5, 48, 40)  # the rest on a small grid (CPU time)
    y = m(cam, rad)
    assert m.features(cam, rad).shape == (B, 24, 48, 40) and y.shape == (B, 7, 48, 40)
    y.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), "gradients reach every parameter"
    n_zero = sum(int(p.grad.abs().sum() == 0) for p in m.parameters())
    assert n_zero == 0, f"{n_zero} parameters without gradient"
    # zero radar: the radar stream is zero, so the first rectify module passes the camera stream unchanged, and the camera
    # alone drives every scale (finite, non-degenerate, gradient to the camera input)
    m.eval()
    f = m.fuser
    with torch.no_grad():
        c0, r0 = f.pc(cam), f.pr(torch.zeros_like(rad))
        assert (r0 == 0).all()
        c1, _ = f.rect[0](c0, r0)
        assert torch.equal(c1, c0), "zero radar: camera stream unchanged"
    cam_g = cam.clone().requires_grad_(True)
    out0 = f(cam_g, torch.zeros_like(rad))
    sum(o.mean() for o in out0).backward()
    assert cam_g.grad.abs().sum() > 0 and all(torch.isfinite(o).all() and o.std() > 1e-3 for o in out0)
    with torch.no_grad():
        out1 = f(cam + 0.5 * torch.randn_like(cam), torch.zeros_like(rad))
    assert all((a - b).abs().mean() > 1e-3 for a, b in zip(out0, out1)), "the camera alone drives every scale"
    # modality dropout only in training
    m2 = CMXTrunk(12, 5, n_classes=7, embed=E, c_out=24, p_drop=0.5).eval()
    with torch.no_grad():
        assert torch.equal(m2(cam, rad), m2(cam, rad))
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.device("meta"):                                # counted without computing
            full = CMXTrunk(80, 32, n_classes=14).eval()
            ci, ri = torch.empty(1, 80, 120, 104), torch.empty(1, 32, 120, 104)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            full(ci, ri)
        flops = (fc.get_total_flops() / 1e9, sum(p.numel() for p in full.parameters()) / 1e6)
    except Exception as e:                                        # the size line is informative only
        print("  (size line skipped:", e, ")")
    return flops


def _selftest():
    rot_err = _test_ego_warp_gru()
    print(f"EgoWarpGRU ok: 1 m forward = exact 2-cell shift with the last 2 rows invalid, pure rotation = splat_positions "
          f"(inverse exact, linear world exact, blob centroid {rot_err * 100:.2f} cm), validity = source inside the old "
          f"grid, two-step pull, state pulled by the step, unknown cells, gradients to every GRU parameter, TemporalFusion "
          f"interface")
    n = _test_lift_s()
    print(f"LiftS ok: table geometry, one-hot depth: S voxel in PD's support and within 1 cell of PD's centre ({n} rays), "
          f"mass conserved (uniform and random depth), rig shift = 2 cells, LiftPD output shape, gradcheck, float32")
    flops = _test_cmx()
    msg = "" if flops is None else (f"; default CMXTrunk (embed 128, 80 + 32 in, 120 x 104): {flops[0]:.1f} GFLOPs, "
                                    f"{flops[1]:.2f} M parameters")
    print(f"CMXFuser ok: four scales 120 x 104 / 60 x 52 / 30 x 26 / 15 x 13, decoder to 0.5 m, head, gradients to every "
          f"parameter, zero radar keeps the camera stream, camera alone drives every scale, no dropout in eval{msg}")


if __name__ == "__main__":
    _selftest()

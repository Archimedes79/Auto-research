"""Loop 10: the learned trace network (arm A of the sparse-graph proposal). Differentiable step network: frozen BEV features -> stem pyramid -> per head: fan + cross-section sampling,
two transformer blocks, step heads (heading bin + residual, step class + residual, offset, on-lane, event, fork angle, merge pointer). Parameter-free operations: bilinear sampling, scatter of the covered crop.
Usage: imported by p10_train.py / p10_infer.py.  Coordinates: metres in the ego frame, (row, column) = (p0, p1); heading theta: u = (cos, sin) in these coordinates."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BIN = math.radians(11.25)
N_BIN = 9
STEPS = torch.tensor([2.0, 4.0, 6.0])
LOOKS = (2.0, 4.0, 6.0)
OFFS = (-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5)
CIN = 42


def smp(maps, pts):
    """maps [B,C,H,W]; pts [B,N,K,2] (p0 = row, p1 = column, metres, +-50) -> [B,N,K,C]"""
    B, N, K, _ = pts.shape
    grid = torch.stack([pts[..., 1] / 50.0, pts[..., 0] / 50.0], -1).reshape(B, N * K, 1, 2).to(maps.dtype)
    s = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return s[..., 0].permute(0, 2, 1).reshape(B, N, K, -1)


def unit(th):
    return torch.stack([torch.cos(th), torch.sin(th)], -1)


def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Stem(nn.Module):
    def __init__(self, cin=CIN):
        super().__init__()
        self.c0 = nn.Sequential(conv_bn(cin, 64), conv_bn(64, 64))
        self.c1 = nn.Sequential(nn.MaxPool2d(2), conv_bn(64, 96), conv_bn(96, 96))
        self.c2 = nn.Sequential(nn.MaxPool2d(2), conv_bn(96, 128), conv_bn(128, 128))
        self.seed = nn.Conv2d(64, 1, 1)
        self.dirh = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        f0 = self.c0(x)
        f1 = self.c1(f0)
        f2 = self.c2(f1)
        return [torch.cat([f0, x[:, 39:42]], 1), f1, f2], self.seed(f0)[:, 0], self.dirh(f0)


class TLayer(nn.Module):
    def __init__(self, d, h, ff=512):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        S, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(S, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(S, T, d)
        x = x + self.o(a)
        return x + self.ff(self.n2(x))


def rot_rel(xy, pos, th):
    """context points xy [B,N,M,2] relative to head position pos [B,N,2] in the heading frame -> (u, v) [B,N,M]"""
    rel = xy - pos[:, :, None, :]
    c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
    return rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c


class TraceNet(nn.Module):
    def __init__(self, d=128, heads=4, l1=3, l2=2, use_cov=True, use_ptr=True, use_range=True, nav=False):
        super().__init__()
        self.d, self.use_cov, self.use_ptr, self.use_range, self.nav = d, use_cov, use_ptr, use_range, nav
        if nav:  # navigation attribute inside the network: SD raster branch (no explicit registration) + route head on the step token
            def gc(i, o, s=1):
                return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

            self.sd_cnn = nn.Sequential(gc(5, 32), gc(32, 48, 2), gc(48, 64, 2), gc(64, 64), gc(64, 64))
            self.route_mlp = nn.Sequential(nn.Linear(d + 64 + len(OFFS) * 5, 128), nn.GELU(), nn.Linear(128, 1))
        self.stem = Stem()
        c_fan = 67 + 96 + 128
        self.fan = nn.Sequential(nn.Linear(c_fan, d), nn.GELU(), nn.Linear(d, d))
        self.fan_pos = nn.Parameter(torch.randn(27, d) * 0.02)
        self.hist = nn.Linear(4, d)
        self.hist_pos = nn.Parameter(torch.randn(6, d) * 0.02)
        self.state = nn.Sequential(nn.Linear(17, d), nn.GELU(), nn.Linear(d, d))
        self.cov = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, d, 4, stride=4))
        self.cov_pos = nn.Parameter(torch.randn(16, d) * 0.02)
        self.l1 = nn.ModuleList([TLayer(d, heads) for _ in range(l1)])
        self.cross = nn.Sequential(nn.Linear(len(OFFS) * 0 + 67 + 96, d), nn.GELU(), nn.Linear(d, d))
        self.cross_pos = nn.Parameter(torch.randn(len(OFFS), d) * 0.02)
        self.l2 = nn.ModuleList([TLayer(d, heads) for _ in range(l2)])
        self.n_out1, self.n_out2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.h_head = nn.Linear(d, N_BIN + 1)
        self.h_step = nn.Linear(d, 3 + 1)
        self.h_off = nn.Linear(d, 1)
        self.h_on = nn.Linear(d, 1)
        self.h_ev = nn.Linear(d, 5)
        self.h_psi = nn.Linear(d, 1)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.ptr_none = nn.Linear(d, 1)
        self.register_buffer("steps", STEPS.clone())
        self.register_buffer("offs", torch.tensor(OFFS))

    # ----------------------------------------------------------------- pieces
    def state_vec(self, pos, th, flag):
        r = pos.norm(dim=-1)
        az = torch.atan2(pos[..., 1], pos[..., 0])
        f = []
        for k in (1, 2, 4, 8):
            f += [torch.sin(2 * math.pi * k * r / 70.0), torch.cos(2 * math.pi * k * r / 70.0)]
        for k in (1, 2):
            f += [torch.sin(k * az), torch.cos(k * az)]
        if not self.use_range:
            f = [x * 0 for x in f]
        f += [pos[..., 0] / 50.0, pos[..., 1] / 50.0, flag, torch.cos(th), torch.sin(th)]
        return torch.stack(f, -1)

    def cov_tokens(self, cv, pos, th, flag):
        B, N, M, _ = cv.shape
        u, v = rot_rel(cv[..., 0:2], pos, th)
        valid = cv[..., 5] > 0.5
        iu, iv = torch.floor(u + 4.0).long(), torch.floor(v + 8.0).long()
        ok = valid & (iu >= 0) & (iu < 16) & (iv >= 0) & (iv < 16)
        idx = (iu.clamp(0, 15) * 16 + iv.clamp(0, 15))
        c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
        cosrel = (cv[..., 2] * c + cv[..., 3] * s) * flag[:, :, None]
        occ = torch.zeros(B, N, 256, device=cv.device, dtype=torch.float32)
        cs = torch.zeros_like(occ)
        occ.scatter_add_(2, idx, ok.float())
        cs.scatter_add_(2, idx, ok.float() * cosrel)
        raster = torch.stack([occ.clamp(max=1.0), (cs / occ.clamp(min=1.0))], 2).reshape(B * N, 2, 16, 16)
        t = self.cov(raster)  # [BN, d, 4, 4]
        return t.flatten(2).transpose(1, 2).reshape(B, N, 16, self.d) + self.cov_pos

    def pointer(self, h2, cv, pos, th, flag):
        B, N, M, _ = cv.shape
        u, v = rot_rel(cv[..., 0:2], pos, th)
        c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
        cosrel = (cv[..., 2] * c + cv[..., 3] * s) * flag[:, :, None]
        sinrel = (-cv[..., 2] * s + cv[..., 3] * c) * flag[:, :, None]
        feat = torch.stack([u / 16.0, v / 16.0, cosrel, sinrel, cv[..., 4]], -1)
        k = self.ptr_k(feat)  # [B,N,M,d]
        q = self.ptr_q(h2)  # [B,N,d]
        lg = (k * q[:, :, None, :]).sum(-1) / math.sqrt(self.d)
        mask = (cv[..., 5] > 0.5) & (cv[..., 4] > 0.5)
        lg = lg.masked_fill(~mask, -1e4)
        none = self.ptr_none(h2)
        return torch.cat([lg, none], -1)  # [B,N,M+1]

    # ----------------------------------------------------------------- forward
    def encode(self, x):
        return self.stem(x)

    def route_logit(self, h2, qfin, cp, sd, detach=False):
        """navigation attribute of the vertex produced by the step: token h2 [B,N,d], vertex position qfin [B,N,2], cross-section positions cp [B,N,7,2], SD raster sd [B,5,200,200] (any pose error / fault)"""
        sf = self.sd_cnn(sd)  # [B,64,50,50]
        f_at = smp(sf, qfin[:, :, None, :])[:, :, 0]
        raw = smp(sd, cp).flatten(2)
        h = h2.detach() if detach else h2
        return self.route_mlp(torch.cat([h, f_at, raw], -1))[..., 0]

    def forward(self, feats, S, qc=None, sd=None, route_detach=False):
        """feats: list of 3 maps from the stem; S: dict of tensors [B,N,...] (pos, theta, flag, hist, cv); qc [B,N,2] cross-section centre (training: expert candidate; None: decode)."""
        pos, th, flag, hist, cv = S["pos"], S["theta"], S["flag"], S["hist"], S["cv"]
        B, N = pos.shape[:2]
        k = (torch.arange(N_BIN, device=pos.device) - N_BIN // 2).float() * BIN
        ang = th[:, :, None, None] + k[None, None, :, None]
        d = torch.tensor(LOOKS, device=pos.device)[None, None, None, :]
        q = pos[:, :, None, None, :] + d[..., None] * unit(ang)
        q = q.reshape(B, N, 27, 2)
        fan = torch.cat([smp(f, q) for f in feats], -1)  # [B,N,27,291]
        head = self.state(self.state_vec(pos, th, flag))[:, :, None]
        toks = [head, self.fan(fan) + self.fan_pos, self.hist(hist) + self.hist_pos]
        if self.use_cov:
            toks.append(self.cov_tokens(cv, pos, th, flag))
        else:
            toks.append(torch.zeros(B, N, 16, self.d, device=pos.device, dtype=head.dtype))
        t = torch.cat(toks, 2).reshape(B * N, -1, self.d)
        for lyr in self.l1:
            t = lyr(t)
        h1 = self.n_out1(t[:, 0]).reshape(B, N, self.d)
        o1 = self.h_head(h1)
        o_step = self.h_step(h1)
        res_phi = torch.tanh(o1[..., N_BIN])
        res_len = 2.0 * torch.tanh(o_step[..., 3])
        out = dict(head=o1[..., :N_BIN], res_phi=res_phi, step=o_step[..., :3], res_len=res_len)
        if qc is None:
            kb = out["head"].argmax(-1)
            cb = out["step"].argmax(-1)
            phi = (kb - N_BIN // 2).float() * BIN + res_phi * math.radians(6.0)
            ln = self.steps[cb] + res_len
            qc = pos + ln[..., None] * unit(th + phi)
            out["qc"] = qc
        chord = qc - pos
        cd = chord / chord.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        nrm = torch.stack([-cd[..., 1], cd[..., 0]], -1)
        cp = qc[:, :, None, :] + self.offs[None, None, :, None] * nrm[:, :, None, :]  # [B,N,7,2]
        cr = torch.cat([smp(feats[0], cp), smp(feats[1], cp)], -1)
        t2 = torch.cat([h1[:, :, None], self.cross(cr) + self.cross_pos, self.hist(hist) + self.hist_pos], 2).reshape(B * N, -1, self.d)
        for lyr in self.l2:
            t2 = lyr(t2)
        h2 = self.n_out2(t2[:, 0]).reshape(B, N, self.d)
        out["off"] = 2.5 * torch.tanh(self.h_off(h2)[..., 0])
        out["on"] = self.h_on(h2)[..., 0]
        out["ev"] = self.h_ev(h2)
        out["psi"] = self.h_psi(h2)[..., 0]
        out["ptr"] = self.pointer(h2, cv, pos, th, flag) if self.use_ptr else torch.zeros(B, N, cv.shape[2] + 1, device=pos.device)
        out["nrm"] = nrm
        out["h2"] = h2
        out["cp"] = cp
        if self.nav and sd is not None:
            out["route"] = self.route_logit(h2, qc + out["off"][..., None] * nrm, cp, sd, route_detach)
        out["qc_used"] = qc
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class MLayer(nn.Module):
    """transformer layer with a key padding mask (graph transformer over the vertex tokens of one frame)"""

    def __init__(self, d, h, ff=256):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :]).transpose(1, 2).reshape(B, T, d)
        x = x + self.o(a)
        return x + self.ff(self.n2(x))


class RouteGraphNet(nn.Module):
    """E103c: navigation attribute as a graph transformer over the vertex tokens of the trace network (frozen tokens h2, positions, cross-section positions) and the SD raster (CNN, no registration).
    Per vertex: [token, SD features at the vertex, raw SD at the 7 cross-section points, position]; scene summary of the SD raster is added to every vertex; attention over all vertices of the frame."""

    def __init__(self, d=96, heads=4, layers=3, d_tok=128):
        super().__init__()

        def gc(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.sd_cnn = nn.Sequential(gc(5, 32), gc(32, 48, 2), gc(48, 64, 2), gc(64, 64), gc(64, 64))
        self.tok = nn.Linear(d_tok, d)
        self.vert = nn.Sequential(nn.Linear(64 + len(OFFS) * 5 + 4 + 64 + 5, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([MLayer(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)

    def forward(self, tok, xy, cp, sd, mask):
        """tok [B,n,128], xy [B,n,2], cp [B,n,7,2], sd [B,5,200,200], mask [B,n] (valid vertices) -> route logits [B,n]"""
        B, n = xy.shape[:2]
        sf = self.sd_cnn(sd)
        f_at = smp(sf, xy[:, :, None, :])[:, :, 0]
        raw = smp(sd, cp).flatten(2)
        scene = torch.cat([sf.mean((2, 3)), sd.mean((2, 3))], -1)[:, None].expand(B, n, -1)
        pe = torch.cat([xy / 50.0, torch.sin(xy.norm(dim=-1, keepdim=True) / 10.0), torch.cos(xy.norm(dim=-1, keepdim=True) / 10.0)], -1)
        x = self.tok(tok) + self.vert(torch.cat([f_at, raw, pe, scene], -1))
        for lyr in self.layers:
            x = lyr(x, mask)
        return self.out(x)[..., 0]


def warp_raster(sd, d, yaw):
    """re-sample the SD raster [B,5,200,200] so that a map drawn with the rigid pose error x -> R(yaw) x + d is brought back into the ego frame: p_in = R^T (p_out - d). d [B,2] metres (p0, p1), yaw [B] radians."""
    B = sd.shape[0]
    dev = sd.device
    ax = (torch.arange(200, device=dev, dtype=torch.float32) + 0.5) * 0.5 - 50.0
    p0, p1 = torch.meshgrid(ax, ax, indexing="ij")
    pout = torch.stack([p0, p1], -1)[None].expand(B, 200, 200, 2) - d[:, None, None, :]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    # R = [[c, -s], [s, c]] acting on column vectors (p0, p1): R^T v = (c v0 + s v1, -s v0 + c v1)
    pin0 = c * pout[..., 0] + s * pout[..., 1]
    pin1 = -s * pout[..., 0] + c * pout[..., 1]
    grid = torch.stack([pin1 / 50.0, pin0 / 50.0], -1)
    return F.grid_sample(sd, grid.to(sd.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)


class RouteLocNet(nn.Module):
    """E103d: navigation attribute with a LEARNED rough localisation: a graph transformer over the vertex tokens and the (mis-posed) SD raster predicts the rigid pose error (translation, yaw) of the map,
    the raster is warped back by the predicted pose (differentiable grid_sample, so the route loss also trains the pose estimate), and a second graph transformer gives the route probability per vertex. No hand-built registration."""

    def __init__(self, d=96, heads=4, layers=3):
        super().__init__()
        self.pose_net = RouteGraphNet(d, heads, 2)
        self.pose_net.out = nn.Linear(d, d)
        self.pose_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        self.route_net = RouteGraphNet(d, heads, layers)

    def forward(self, tok, xy, cp, sd, mask, pose_in=None):
        B, n = xy.shape[:2]
        # stage A: pose from the vertex tokens (pose_net returns the per-vertex features of its transformer)
        x = self.pose_net.tok(tok) + self.pose_net.vert(self._vfeat(self.pose_net, tok, xy, cp, sd))
        for lyr in self.pose_net.layers:
            x = lyr(x, mask)
        m = mask[..., None].float()
        pooled = torch.cat([(x * m).sum(1) / m.sum(1).clamp(min=1), (x.masked_fill(~mask[..., None], -1e4)).max(1).values], -1)
        o = self.pose_head(pooled)
        d = 4.0 * torch.tanh(o[:, :2])
        yaw = math.radians(8.0) * torch.tanh(o[:, 2])
        if pose_in is not None:  # teacher forcing of the pose (training stage B on the true pose)
            d, yaw = pose_in
        sdw = warp_raster(sd, d, yaw)
        # recompute the cross-section samples are in ego frame already; stage B on the warped raster
        lg = self.route_net(tok, xy, cp, sdw, mask)
        return lg, d, yaw

    @staticmethod
    def _vfeat(net, tok, xy, cp, sd):
        B, n = xy.shape[:2]
        sf = net.sd_cnn(sd)
        f_at = smp(sf, xy[:, :, None, :])[:, :, 0]
        raw = smp(sd, cp).flatten(2)
        scene = torch.cat([sf.mean((2, 3)), sd.mean((2, 3))], -1)[:, None].expand(B, n, -1)
        pe = torch.cat([xy / 50.0, torch.sin(xy.norm(dim=-1, keepdim=True) / 10.0), torch.cos(xy.norm(dim=-1, keepdim=True) / 10.0)], -1)
        return torch.cat([f_at, raw, pe, scene], -1)


def warp_generic(img, d, yaw, size):
    """like warp_raster for rasters of `size` cells covering +-50 m"""
    B = img.shape[0]
    dev = img.device
    cell = 100.0 / size
    ax = (torch.arange(size, device=dev, dtype=torch.float32) + 0.5) * cell - 50.0
    p0, p1 = torch.meshgrid(ax, ax, indexing="ij")
    pout = torch.stack([p0, p1], -1)[None].expand(B, size, size, 2) - d[:, None, None, :]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    pin0 = c * pout[..., 0] + s * pout[..., 1]
    pin1 = -s * pout[..., 0] + c * pout[..., 1]
    grid = torch.stack([pin1 / 50.0, pin0 / 50.0], -1)
    return F.grid_sample(img, grid.to(img.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)


class CorrLoc(nn.Module):
    """Differentiable rough localisation by a correlation volume (translation +-4 m in 1 m steps, yaw +-8 deg in 2 deg steps) between a raster of the traced graph (Gaussian splats of the vertices) and the road raster of the
    SD map (blurred), followed by a soft-argmax. The only parameters are the softmax temperature and the splat width; the pose is differentiable with respect to the vertex positions and the SD raster."""

    def __init__(self, sigma=3.0, tau=60.0, n_shift=4, yaws=(-8, -6, -4, -2, 0, 2, 4, 6, 8)):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.sigma, self.ns = sigma, n_shift
        self.register_buffer("yaws", torch.deg2rad(torch.tensor(yaws, dtype=torch.float32)))

    def graph_raster(self, xy, mask, size=100):
        B, n, _ = xy.shape
        cell = 100.0 / size
        ax = (torch.arange(size, device=xy.device, dtype=torch.float32) + 0.5) * cell - 50.0
        p0, p1 = torch.meshgrid(ax, ax, indexing="ij")
        grid = torch.stack([p0, p1], -1).reshape(1, size * size, 1, 2)
        d2 = ((grid - xy[:, None, :, :]) ** 2).sum(-1)  # [B,size*size,n]
        w = torch.exp(-d2 / (2 * self.sigma ** 2)) * mask[:, None, :].float()
        return w.sum(-1).reshape(B, 1, size, size)

    def forward(self, xy, mask, sd):
        B = xy.shape[0]
        size = 100
        G = self.graph_raster(xy, mask, size)
        S = F.avg_pool2d(sd[:, 1:2].float(), 2)  # roads channel at 1 m
        k = 7
        g1 = torch.exp(-(torch.arange(k, device=sd.device, dtype=torch.float32) - k // 2) ** 2 / (2 * (self.sigma / 1.0) ** 2))
        ker = (g1[:, None] * g1[None, :])[None, None]
        ker = ker / ker.sum()
        S = F.conv2d(S, ker, padding=k // 2)
        R = len(self.yaws)
        Sr = torch.stack([warp_generic(S, torch.zeros(B, 2, device=sd.device), self.yaws[i].expand(B), size) for i in range(R)], 1)  # [B,R,1,size,size]
        Sr = Sr.reshape(B * R, 1, size, size)
        Sr = (Sr - Sr.mean((2, 3), keepdim=True)) / (Sr.flatten(1).norm(dim=1)[:, None, None, None] + 1e-4)
        Gn = (G - G.mean((2, 3), keepdim=True)) / (G.flatten(1).norm(dim=1)[:, None, None, None] + 1e-4)
        ns = self.ns
        inp = F.pad(Sr, (ns, ns, ns, ns)).reshape(1, B * R, size + 2 * ns, size + 2 * ns)
        wgt = Gn[:, None].expand(B, R, 1, size, size).reshape(B * R, 1, size, size)
        C = F.conv2d(inp, wgt, groups=B * R).reshape(B, R, 2 * ns + 1, 2 * ns + 1)
        P = torch.softmax(C.flatten(1) * self.log_tau.exp(), -1).reshape(B, R, 2 * ns + 1, 2 * ns + 1)
        sh = torch.arange(-ns, ns + 1, device=sd.device, dtype=torch.float32)
        s0 = (P.sum((1, 3)) * sh[None]).sum(-1)  # expected shift along p0 (cells of 1 m)
        s1 = (P.sum((1, 2)) * sh[None]).sum(-1)
        yaw = -(P.sum((2, 3)) * self.yaws[None]).sum(-1)
        valid = (mask.sum(1) > 0) & (sd[:, 1].flatten(1).sum(1) > 0)
        d = torch.stack([s0, s1], -1) * (100.0 / size)
        d = torch.where(valid[:, None], d, torch.zeros_like(d))
        yaw = torch.where(valid, yaw, torch.zeros_like(yaw))
        return d, yaw


class RouteCorrNet(nn.Module):
    """E103e: route attribute with a differentiable correlation-volume localisation (CorrLoc) in front of the graph transformer route head."""

    def __init__(self, d=96, heads=4, layers=3):
        super().__init__()
        self.loc = CorrLoc()
        self.route_net = RouteGraphNet(d, heads, layers)

    def forward(self, tok, xy, cp, sd, mask, pose_in=None):
        dd, yaw = self.loc(xy, mask, sd.float())
        if pose_in is not None:
            dd, yaw = pose_in
        sdw = warp_raster(sd, dd, yaw)
        # the cross-section samples were taken in the ego frame: the raw SD samples are recomputed on the warped raster inside route_net
        return self.route_net(tok, xy, cp, sdw, mask), dd, yaw


def fourier2d(xy, n_freq=4):
    """2-D Fourier features of metric positions (+-50 m): [..., 2] -> [..., 4 * n_freq]"""
    f = (2.0 ** torch.arange(n_freq, device=xy.device, dtype=xy.dtype)) * math.pi / 64.0
    a = xy[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class XLayer(nn.Module):
    """cross-attention layer: the list `x` (vertex tokens) pulls information from another list `mem` (map tokens); the raw input `x0` of the vertex is fused with the attention result (gated residual), then a feed-forward block"""

    def __init__(self, d, h, ff=256):
        super().__init__()
        self.h = h
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, x0, mem):
        B, n, d = x.shape
        T = mem.shape[1]
        q = self.q(self.nq(x)).view(B, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(self.nk(mem)).view(B, T, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, n, d)
        g = self.fuse(torch.cat([x, x0], -1))  # how much of the raw input is kept next to the attention result
        x = x + self.o(a) + g * x0
        return x + self.ff(self.n2(x))


class RouteXNet(nn.Module):
    """E104: navigation attribute with cross-attention: the vertex tokens (self-attention over the list, E103c) pull information from the list of SD-map tokens (CNN features of the SD raster at 2 m cells with Fourier position codes) by cross-attention with a global receptive field, and fuse it with their own input; no registration step."""

    def __init__(self, d=96, heads=4, layers=3, d_tok=128):
        super().__init__()

        def gc(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.sd_cnn = nn.Sequential(gc(5, 32), gc(32, 48, 2), gc(48, 64, 2), gc(64, 64))
        self.map_proj = nn.Linear(64 + 16, d)
        self.tok = nn.Linear(d_tok, d)
        self.vert = nn.Sequential(nn.Linear(16 + len(OFFS) * 5 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.self_layers = nn.ModuleList([MLayer(d, heads) for _ in range(layers)])
        self.x_layers = nn.ModuleList([XLayer(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        ax = (torch.arange(50, dtype=torch.float32) + 0.5) * 2.0 - 50.0
        p0, p1 = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("cell_xy", torch.stack([p0, p1], -1).reshape(1, 2500, 2))

    def forward(self, tok, xy, cp, sd, mask):
        B, n = xy.shape[:2]
        sf = self.sd_cnn(sd)  # [B,64,50,50]
        mem = self.map_proj(torch.cat([sf.flatten(2).transpose(1, 2), fourier2d(self.cell_xy.expand(B, -1, -1))], -1))
        raw = smp(sd, cp).flatten(2)
        pe = torch.cat([xy / 50.0, torch.sin(xy.norm(dim=-1, keepdim=True) / 10.0), torch.cos(xy.norm(dim=-1, keepdim=True) / 10.0)], -1)
        x0 = self.tok(tok) + self.vert(torch.cat([fourier2d(xy), raw, pe], -1))
        x = x0
        for sl, xl in zip(self.self_layers, self.x_layers):
            x = sl(x, mask)
            x = xl(x, x0, mem)
        return self.out(x)[..., 0]

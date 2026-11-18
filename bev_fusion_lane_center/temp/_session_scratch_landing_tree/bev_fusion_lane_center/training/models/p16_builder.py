"""Loop 16: the unified differentiable lane-graph builder (designs D1 - D8 of the L8 array) - one module, switches A - G.
A construction G (grow from tips: verified-ray hypothesis tree) | L (spray, snap, link: one reading round of bidirectional tips + link head)
B pattern 'fixed' (E94-like fan, expectation read-out, one child) | 'teardrop' (user sketch: survival read-out, b best endpoints)
C seeds 'lattice' | 'anchors'     D dedup 'geo' | 'learned' (pair head)     E context 'local' | 'global'     F warm start (scheduled expert states)     G revisit (E120-style layers)
Input: frozen cached BEV channels (42) -> stem (initialised from E94) -> pyramid. Output (dict): nodes with position, validity, path validity, parent, level, flag, token; edges parent -> child; seeds; links.
Forbidden on the training path: argmax / hard top-k / NMS on predicted quantities except in the functions marked SELECTION (index choice; the gradient flows through the selected items and their scores)."""
import bflc_paths as BP  # project paths from the location of this checkout
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import p10_net as PN
import p16_geom as PG

BUCKETS = torch.tensor([1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0, 20.0, 30.0])
_CONST = {}


def const(vals, device):
    """small constant tensors on the device, created once (a torch.tensor(..., device=) in the forward is a host-to-device copy: it blocks CUDA-graph capture)"""
    k = (tuple(vals), str(device))
    if k not in _CONST:
        _CONST[k] = torch.tensor(list(vals), device=device)
    return _CONST[k]


def samp(maps, q):
    """bilinear sampling of maps [B,C,H,W] (50 m around the ego) at points q [B,...,2] (p0 = row, p1 = column, metres) -> [B,...,C]"""
    B = q.shape[0]
    lead = q.shape[1:-1]
    flat = q.reshape(B, -1, 1, 2)
    grid = torch.stack([flat[..., 1] / 50.0, flat[..., 0] / 50.0], -1).to(maps.dtype)
    s = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return s[..., 0].permute(0, 2, 1).reshape(B, *lead, -1)


def ring_points(g, a):
    """seed candidates on the square ring at +-a: the positions g on each of its four sides and the four corners -> [4 len(g) + 4, 2]"""
    s, c = torch.full_like(g, a), const((-a, a), g.device)
    corners = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
    return torch.cat([torch.stack([-s, g], -1), torch.stack([s, g], -1), torch.stack([g, -s], -1), torch.stack([g, s], -1), corners])


def unitv(th):
    return torch.stack([torch.cos(th), torch.sin(th)], -1)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


def grad_scale(x, s):
    """identity in the forward pass, the gradient is multiplied by s (s = 0 detaches, s = 1 does nothing)"""
    return x if s == 1.0 else _GradScale.apply(x, s)


def hard_concrete(logit, training, beta=2.0 / 3.0, gamma=-0.1, zeta=1.1):
    """stretched and clipped concrete gate (Louizos et al.): sampled in training, deterministic at inference; returns the gate z in [0,1] and P(z > 0) (expected L0)"""
    if training:
        u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logit) / beta)
    else:
        s = torch.sigmoid(logit)
    z = (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)
    return z, torch.sigmoid(logit - beta * math.log(-gamma / zeta))


def gather1(t, idx):
    """t [B,S,*rest], idx [B,K] -> [B,K,*rest]"""
    rest = t.shape[2:]
    return t.gather(1, idx.reshape(idx.shape[0], idx.shape[1], *([1] * len(rest))).expand(idx.shape[0], idx.shape[1], *rest))


def gflat(t, sel):
    """t [B,Nn,b,*rest] flattened over (Nn,b) and gathered by sel [B,M] -> [B,M,*rest]"""
    B = t.shape[0]
    return gather1(t.reshape(B, -1, *t.shape[3:]), sel)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class DefaultCfg:
    constr = "G"          # A: G | L | H (hybrid: tree of D = 3 levels + link head on the ends of the last level)
    pattern = "teardrop"  # B: teardrop | fixed
    seeds = "lattice"     # C: lattice | anchors
    dedup = "geo"         # D: geo | learned
    ctx = "local"         # E: local | global
    warm = False          # F
    revisit = False       # G
    d = 96
    b = 2                 # children per tip (teardrop); fixed pattern: 1
    D = None              # levels: teardrop 4, fixed 8, L 1
    M = 96                # beam: tips kept per level
    K0 = 64               # seeds kept
    lattice = 5.0         # spacing (m) of the seed lattice
    lattice_ring = 0.0    # E266: > 0 adds the square ring of seed candidates at +-lattice_ring m at inference (0 = off)
    n_anchor = 128
    tolerant = False
    stage_b = False       # HF1: refine children
    stage_c = False       # HF2: rescore children
    spf2 = False          # SPF2: learned keep gate (hard concrete) on every node
    reach = None          # SPF3: reach of the ray ladder in metres (teardrop): 24 default, 12 short
    mcl = False           # HF3: multiple-choice learning of the children (loss side)
    junc = False          # junction head: split / merge heat maps (E44-type key points), trained against the ground-truth landmarks; used by the assembly (rung 3)
    fork_relax = False    # loop 16b: children selection relaxed in front of a predicted fork (junction heat ahead of the tip), shallow forks keep their second branch
    relax_thr = 0.4
    relax_r = 0.6
    relax_seg = 0.35
    temporal = False      # T.2: one-step time recursion (previous-frame nodes as seeds, log-odds fusion), rung 3
    detach_pos = False    # flip (VS19 / VS21): stop the gradient on the coordinates that decide the next read (RAFT form), tokens keep their gradient
    seed_gscale = 1.0     # gradient scale of the seed state (position, heading, token) entering the tree; 1 = none (S stabilise uses 0.1)
    ver_mode = "survival" # verification read-out: "survival" (spec) | "loi" (baseline B3, one pooled score per ray)
    link_mode = "learned" # link head: "learned" | "geo" (baseline B1, geometric prior only)
    pi_mode = "min"       # path validity: "min" (bottleneck, spec) | "prod" (product of the node validities; beam papers sum log scores, MH41)
    local_r = 12.0
    heads = 4

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)
        if self.D is None:
            self.D = 1 if self.constr == "L" else (3 if self.constr == "H" else (4 if self.pattern == "teardrop" else 8))
        if self.pattern == "fixed":
            self.b = 1

    def asdict(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith("_") and not callable(getattr(self, k))}


class CtxLayer(nn.Module):
    """graph-biased self-attention over tip tokens (queries) and a key set (tips, optionally all committed nodes): bias per head from the distance bucket; local = keys within a radius only"""

    def __init__(self, d, heads=4, local_r=None):
        super().__init__()
        self.h, self.local_r = heads, local_r
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.bias = nn.Embedding(len(BUCKETS) + 1, heads)
        self.ff = mlp(d, 2 * d, d)
        self.register_buffer("bk", BUCKETS.clone())

    def forward(self, h, pos, valid, kh, kpos, kvalid):
        B, N, d = h.shape
        T = kh.shape[1]
        q = self.q(self.n1(h)).view(B, N, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(self.n1(kh)).view(B, T, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        dist = torch.cdist(pos.float(), kpos.float())
        bias = self.bias(torch.bucketize(dist, self.bk)).permute(0, 3, 1, 2).to(q.dtype)
        m = (valid[:, None, :, None] > 0) & (kvalid[:, None, None, :] > 0)
        if self.local_r is not None:
            m = m & (dist[:, None] <= self.local_r)
        m = m | torch.eye(N, T, dtype=torch.bool, device=h.device)[None, None] & (valid[:, None, :, None] > 0)  # a tip always sees itself (keys start with the tips)
        bias = bias.masked_fill(~m, -1e4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(B, N, d)
        h = h + self.o(a)
        return h + self.ff(self.n2(h))


class Builder(nn.Module):
    def __init__(self, cfg=None, stem_from="e94_a", anchors=None):
        super().__init__()
        self.cfg = cfg or DefaultCfg()
        c = self.cfg
        d = c.d
        self.stem = PN.Stem()
        if stem_from:
            sd = torch.load(f"{BP.ROOT}/runs/{stem_from}/model.pt", map_location="cpu")
            self.stem.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("stem.")})
        self.pr = nn.ModuleList([nn.Conv2d(67, 32, 1), nn.Conv2d(96, 32, 1), nn.Conv2d(128, 16, 1)])
        self.nf = 80
        self.pat = PG.Pattern(c.pattern, reach=c.reach)
        R, K = self.pat.R, self.pat.K
        # seeds
        self.n_off = 25
        self.register_buffer("seed_off", torch.tensor([[u, v] for u in (-4.0, -2.0, 0.0, 2.0, 4.0) for v in (-4.0, -2.0, 0.0, 2.0, 4.0)]))
        self.seed_mlp = nn.Sequential(nn.Linear(self.n_off * self.nf + 12 + 2, 192), nn.GELU(), nn.Linear(192, d), nn.GELU())
        self.seed_val, self.seed_snap, self.seed_dir = nn.Linear(d, 1), nn.Linear(d, 2), nn.Linear(d, 2)
        nn.init.zeros_(self.seed_snap.weight), nn.init.zeros_(self.seed_snap.bias), nn.init.zeros_(self.seed_dir.weight), nn.init.zeros_(self.seed_dir.bias)
        self.flag_emb = nn.Embedding(2, d)
        if c.seeds == "anchors":
            self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32) if anchors is not None else torch.zeros(c.n_anchor, 4))
        # context
        self.ctx = CtxLayer(d, c.heads, c.local_r if c.ctx == "local" else None)
        # ray reader
        self.geo_emb = nn.Parameter(torch.randn(R, K, 16) * 0.1)
        self.hp = nn.Linear(d, 48)
        self.read = nn.Sequential(nn.Linear(self.nf + 16 + 48, 96), nn.GELU(), nn.Linear(96, 96))
        self.read_conv = nn.Conv1d(96, 96, 3, padding=1)
        self.read_act = nn.GELU()
        self.out_v, self.out_off, self.out_emb = nn.Linear(96, 1), nn.Linear(96, 2), nn.Linear(96, 32)
        nn.init.zeros_(self.out_off.weight), nn.init.zeros_(self.out_off.bias)
        nn.init.constant_(self.out_v.bias, 1.0)
        if c.pattern == "fixed":
            self.exist = nn.Linear(96, 1)
        if c.ver_mode == "loi":
            self.loi_head = nn.Linear(96, 1)
        if c.junc:
            self.junc_head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, 1))
        self.child = nn.Sequential(nn.Linear(d + 32 + 16 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.child_norm = nn.LayerNorm(d)
        if c.stage_b:
            self.register_buffer("rf_off", torch.tensor([[0.0, 0.0]] + [[1.5 * math.cos(a), 1.5 * math.sin(a)] for a in [k * math.pi / 4 for k in range(8)]]))
            self.rf_mlp = nn.Sequential(nn.Linear(9 * self.nf + d, 128), nn.GELU(), nn.Linear(128, 2 + 32))
            nn.init.zeros_(self.rf_mlp[-1].weight[:2]), nn.init.zeros_(self.rf_mlp[-1].bias[:2])
            self.rf_tok, self.rf_norm = nn.Linear(32, d), nn.LayerNorm(d)
        if c.stage_c:
            self.rc_layer = PN.TLayer(d, c.heads)
            self.rc_head = nn.Linear(d, 1)
            nn.init.zeros_(self.rc_head.weight), nn.init.zeros_(self.rc_head.bias)
        if c.spf2:
            self.keep = nn.Linear(d, 1)
            nn.init.zeros_(self.keep.weight), nn.init.constant_(self.keep.bias, 2.0)
        self.pair = mlp(2 * 32 + 5, 64, 1) if c.dedup == "learned" else None
        self.pair_proj = nn.Linear(d, 32) if c.dedup == "learned" else None
        if c.revisit:
            import p15_refine as PR

            self.rev = nn.ModuleList([PR.RLayer(d, du_max=1.5) for _ in range(2)])
            self.rev_in = nn.Linear(d, d)
        if c.constr in ("L", "H"):
            self.link = mlp(d + d + 9, 96, 1)
            self.conn_v = mlp(self.nf + 4, 48, 1)

    # ---------------------------------------------------------------- features
    def encode(self, x):
        feats, seed_lg, dirh = self.stem(x)
        maps = [p(f) for p, f in zip(self.pr, feats)]
        raw_dir = feats[0][:, 65:67]
        return feats, maps, seed_lg, dirh, raw_dir

    def sample(self, maps, q):
        return torch.cat([samp(m, q) for m in maps], -1)

    # ---------------------------------------------------------------- seeds
    def seed_candidates(self, B, device):
        c = self.cfg
        if c.seeds == "lattice":
            g = torch.arange(-45.0, 45.01, c.lattice, device=device)
            P = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
            if c.lattice_ring > 0 and not self.training:  # E266: the training jitter carries seeds to +-47.5 m, the inference lattice stops at +-45
                P = torch.cat([P, ring_points(g, c.lattice_ring)])  # appended: the lattice candidates keep their index
            P = P[None].expand(B, -1, -1)
            if self.training:
                P = P + (torch.rand_like(P) - 0.5) * c.lattice
            return P, None
        a = self.anchors.to(device)
        return a[None, :, :2].expand(B, -1, -1), torch.atan2(a[:, 3], a[:, 2])[None].expand(B, -1)

    def seeds(self, maps, raw_dir):
        B = raw_dir.shape[0]
        dev = raw_dir.device
        cpos, ath = self.seed_candidates(B, dev)
        S = cpos.shape[1]
        q = cpos[:, :, None, :] + self.seed_off[None, None]
        g = self.sample(maps, q).reshape(B, S, -1)
        r = cpos.norm(dim=-1)
        pe = torch.stack([torch.sin(2 * math.pi * k * r / 70.0) for k in (1, 2, 4, 8)] + [torch.cos(2 * math.pi * k * r / 70.0) for k in (1, 2, 4, 8)] + [cpos[..., 0] / 50.0, cpos[..., 1] / 50.0, r / 50.0, r * 0], -1)
        ah = torch.zeros(B, S, 2, device=dev) if ath is None else unitv(ath)
        h = self.seed_mlp(torch.cat([g.float(), pe, ah], -1).to(g.dtype))
        val = self.seed_val(h)[..., 0].float()
        snap = 3.5 * torch.tanh(self.seed_snap(h).float())
        d0 = samp(raw_dir, cpos).float()
        dvec = d0 + 0.5 * self.seed_dir(h).float()
        th = torch.atan2(dvec[..., 1], dvec[..., 0])
        pos = cpos + snap
        return dict(cpos=cpos, pos=pos, th=th, val=val, h=h, dvec=dvec)

    @staticmethod
    def seeds_select(s, K0, nms_r=3.0):
        """SELECTION: greedy distance NMS on the snapped seed positions by validity, K0 seeds -> indices [B,K0], ok [B,K0]"""
        pos, score = s["pos"].detach(), s["val"].detach()
        B, S, _ = pos.shape
        dist = torch.cdist(pos, pos)
        sc = score.clone()
        idx, ok = [], []
        for _ in range(K0):
            best = sc.argmax(-1)
            o = sc.gather(-1, best[:, None])[:, 0] > -1e8
            idx.append(best)
            ok.append(o)
            sup = dist.gather(1, best[:, None, None].expand(B, 1, S))[:, 0] <= nms_r
            sc = sc.masked_fill(sup, -1e9)
        return torch.stack(idx, 1), torch.stack(ok, 1)

    # ---------------------------------------------------------------- one growth level
    def read_rays(self, maps, pos, th, h):
        """sample the pattern around every tip, verification and offsets per sample.  returns q [B,N,R,K,2], v_logit [B,N,R,K], off [B,N,R,K,2], emb [B,N,R,K,32], z [B,N,R,K,96]"""
        B, N = pos.shape[:2]
        pat = self.pat
        q = PG.ray_points(pos, th, pat)
        S = self.sample(maps, q)
        R, K = pat.R, pat.K
        hp = self.hp(h)[:, :, None, None, :].expand(B, N, R, K, 48)
        geo = self.geo_emb.to(S.dtype)[None, None].expand(B, N, R, K, 16)
        z = self.read(torch.cat([S, geo, hp.to(S.dtype)], -1))
        zc = self.read_conv(z.reshape(B * N * R, K, 96).transpose(1, 2)).transpose(1, 2).reshape(B, N, R, K, 96)
        z = self.read_act(z + zc)
        v_logit = self.out_v(z)[..., 0].float()
        off = self.out_off(z).float()
        emb = self.out_emb(z)
        return q, v_logit, off, emb, z

    def level(self, maps, st, kv, it_share, gt, relax=None):
        """expand every tip of the state `st` into children; returns the children (before the beam) and the training tensors of this level"""
        c, pat = self.cfg, self.pat
        pos, th, h, valid = st["pos"], st["th"], st["h"], st["valid"]
        B, N = pos.shape[:2]
        h = self.ctx(h, pos, valid, kv["h"], kv["pos"], kv["valid"])
        q, v_logit, off, emb, z = self.read_rays(maps, pos, th, h)
        mask = pat.mask
        ray_d = unitv(th[:, :, None] + pat.phi)  # [B,N,R,2]
        ray_n = torch.stack([-ray_d[..., 1], ray_d[..., 0]], -1)
        along, across = 1.5 * torch.tanh(off[..., 0]), 2.0 * torch.tanh(off[..., 1])
        pt = q + along[..., None] * ray_d[:, :, :, None, :] + across[..., None] * ray_n[:, :, :, None, :]  # corrected samples [B,N,R,K,2]
        v = torch.sigmoid(v_logit).clamp(PG.CFG["v_floor"], PG.CFG["v_ceil"]) * mask
        out = dict(q=q, v_logit=v_logit, ray_d=ray_d, pt=pt)
        if c.pattern == "fixed":
            sc = v_logit.masked_fill(mask < 0.5, -1e4).reshape(B, N, -1)
            w = torch.softmax(sc, -1)
            end = (w[..., None] * pt.reshape(B, N, -1, 2)).sum(2)[:, :, None, :]  # [B,N,1,2] expectation read-out
            zc = (w[..., None] * z.reshape(B, N, -1, z.shape[-1])).sum(2)
            nu = torch.sigmoid(self.exist(zc).float())[..., 0][:, :, None]
            emb_c = (w[..., None] * emb.reshape(B, N, -1, 32)).sum(2)[:, :, None, :]
            geo_c = torch.zeros(B, N, 1, 16, device=pos.device, dtype=emb_c.dtype)
            ok = valid[:, :, None] > 0
            nu_sel = nu
            ray_i = torch.zeros(B, N, 1, dtype=torch.long, device=pos.device)
            out["w"] = w
        else:
            kstar, nu_p, T = PG.candidates(v.float(), pat, c.tolerant)
            nu_val = T.gather(-1, kstar[..., None])[..., 0]  # survival to the chosen sample (without the direction prior)
            if c.ver_mode == "loi":  # baseline B3 (K40): ONE score per ray from the pooled samples (line-of-interest verification), the endpoint is the end of the ray
                zr = (z * pat.mask[None, None, :, :, None]).sum(3) / pat.mask.sum(1)[None, None, :, None]
                s_ray = torch.sigmoid(self.loi_head(zr)[..., 0].float()).clamp(PG.CFG["v_floor"], PG.CFG["v_ceil"])
                kstar = (pat.mask.sum(1) - 1).long()[None, None, :].expand(B, N, -1)
                nu_val, nu_p = s_ray, s_ray * pat.prior
                out["loi_logit"] = torch.logit(s_ray)
                T = T * 0 + s_ray[..., None]
            endp = pt.gather(3, kstar[..., None, None].expand(B, N, pat.R, 1, 2))[:, :, :, 0]  # [B,N,R,2]
            nu_sel = nu_p
            if "path" in st:
                cov = PG.covered(endp.detach(), st["path"], st["pmask"])
                nu_sel = torch.where(cov, nu_sel * 0.02, nu_sel)
            idx, okk = PG.select_children(endp.detach(), nu_sel.detach(), pos.detach(), c.b, relax=relax, relax_seg=c.relax_seg, relax_r=c.relax_r)  # SELECTION
            gi = idx[..., None]
            end = endp.gather(2, gi.expand(B, N, c.b, 2))
            nu = nu_val.gather(2, idx)
            ok = okk & (valid[:, :, None] > 0)
            R_ = pat.R
            ks5 = kstar[..., None, None]
            emb_k = emb.gather(3, ks5.expand(B, N, R_, 1, 32))[:, :, :, 0]
            emb_c = emb_k.gather(2, idx[..., None].expand(B, N, c.b, 32))
            geo_k = self.geo_emb.to(emb.dtype)[None, None].expand(B, N, R_, pat.K, 16).gather(3, ks5.expand(B, N, R_, 1, 16))[:, :, :, 0]
            geo_c = geo_k.gather(2, idx[..., None].expand(B, N, c.b, 16))
            ray_i = idx
            out.update(T=T, kstar=kstar, nu_all=nu_val)
        out.update(end=end, nu=nu, ok=ok, ray_i=ray_i)
        b = end.shape[2]
        chord = end - pos[:, :, None, :]
        th_c = torch.atan2(chord[..., 1], chord[..., 0])
        ln = chord.norm(dim=-1, keepdim=True)
        hc = self.child(torch.cat([h[:, :, None, :].expand(B, N, b, h.shape[-1]), emb_c.to(h.dtype), geo_c.to(h.dtype), ln / 24.0], -1))
        hc = self.child_norm(hc + h[:, :, None, :])
        if c.stage_b:  # HF1: refine every child from a small pattern around it (no detach between the stages)
            end, hc = self.refine_children(maps, pos, end, hc)
            out["end"] = end
            chord = end - pos[:, :, None, :]
            th_c = torch.atan2(chord[..., 1], chord[..., 0])
        if c.stage_c:  # HF2: rescore the validity of the children with attention over the parent and the siblings
            nu = self.rescore(h, hc, nu)
            out["nu"] = nu
        return h, out, th_c, hc

    def fork_relax_mask(self, jprob, pos, th, flag):
        """tips in front of a predicted fork: the split heat (forward tips) / merge heat (backward tips) at 3 - 12 m ahead of the tip in the growth direction exceeds relax_thr (detached selection input)"""
        d = unitv(th)
        ds = const([3.0, 6.0, 9.0, 12.0], pos.device)
        pts = pos[:, :, None, :] + ds[None, None, :, None] * d[:, :, None, :]
        hs = samp(jprob[:, 0:1], pts)[..., 0]
        hm = samp(jprob[:, 1:2], pts)[..., 0]
        h = torch.where(flag[..., None] > 0, hs, hm).amax(-1)
        return (h > self.cfg.relax_thr).detach()

    def refine_children(self, maps, pos, end, hc):
        B, N, b, _ = end.shape
        chord = end - pos[:, :, None, :]
        u = F.normalize(chord, dim=-1, eps=1e-6)
        n = torch.stack([-u[..., 1], u[..., 0]], -1)
        off = self.rf_off.to(end.dtype)
        q = end[:, :, :, None, :] + off[:, 0][None, None, None, :, None] * u[:, :, :, None, :] + off[:, 1][None, None, None, :, None] * n[:, :, :, None, :]
        S = self.sample(maps, q).reshape(B, N, b, -1)
        z = self.rf_mlp(torch.cat([S, hc.to(S.dtype)], -1))
        along, across = 1.5 * torch.tanh(z[..., 0].float()), 2.0 * torch.tanh(z[..., 1].float())
        end2 = end + along[..., None] * u + across[..., None] * n
        hc2 = self.rf_norm(hc + self.rf_tok(z[..., 2:]).to(hc.dtype))
        return end2, hc2

    def rescore(self, h, hc, nu):
        B, N, b, d = hc.shape
        seq = torch.cat([h[:, :, None, :], hc], 2).reshape(B * N, 1 + b, d)
        o = self.rc_layer(seq)
        delta = self.rc_head(o[:, 1:]).reshape(B, N, b).float()
        return torch.sigmoid(torch.logit(nu.float().clamp(1e-4, 1 - 1e-4)) + delta)

    # ---------------------------------------------------------------- tree
    @staticmethod
    def warp_prior(prior, pose):
        """previous-frame output (pos, th travel heading in the previous ego frame) -> current ego frame by the ego motion: p_cur = R_c^T (R_p p_prev + t_p - t_c).  pose: Rp, Rc [B,2,2], tp, tc [B,2]"""
        Rp, Rc, tp, tc = pose["Rp"], pose["Rc"], pose["tp"], pose["tc"]
        g = torch.einsum("bij,bpj->bpi", Rp, prior["pos"]) + (tp - tc)[:, None, :]
        p_cur = torch.einsum("bji,bpj->bpi", Rc, g)
        u = torch.stack([torch.cos(prior["th"]), torch.sin(prior["th"])], -1)
        u_cur = torch.einsum("bji,bpj->bpi", Rc, torch.einsum("bij,bpj->bpi", Rp, u))
        return dict(pos=p_cur, th=torch.atan2(u_cur[..., 1], u_cur[..., 0]), pi=prior["pi"], h=prior["h"], valid=prior["valid"])

    def add_prior(self, sd, prior, decay=0.9):
        """the nodes of the previous frame become additional seed candidates (their validity decays); selection and non-maximum suppression treat them like the sprayed seeds"""
        val_p = torch.logit((prior["pi"] * decay).clamp(1e-4, 1 - 1e-4)).float().masked_fill(prior["valid"] < 0.5, -1e4)
        return dict(pos=torch.cat([sd["pos"], prior["pos"]], 1), th=torch.cat([sd["th"], prior["th"]], 1), h=torch.cat([sd["h"], prior["h"].to(sd["h"].dtype)], 1), val=torch.cat([sd["val"], val_p], 1))

    @staticmethod
    def fuse_prior(res, prior, a=0.9, b=1.0, decay=0.9, match_r=1.5, match_deg=30.0):
        """log-odds fusion of the warped previous validity and the current evidence at the nodes that coincide with a previous node (distance and travel heading); the previous side carries no gradient"""
        d = torch.cdist(res["pos"].float(), prior["pos"].float())
        dth = torch.atan2(torch.sin(res["th_travel"][:, :, None] - prior["th"][:, None, :]), torch.cos(res["th_travel"][:, :, None] - prior["th"][:, None, :])).abs()
        ok = (d < match_r) & (dth < math.radians(match_deg)) & (prior["valid"][:, None, :] > 0.5)
        dm, im = d.masked_fill(~ok, 1e9).min(-1)
        matched = dm < 1e8
        pp = gather1(prior["pi"], im.reshape(im.shape[0], -1)).reshape(im.shape) * decay
        lg = a * torch.logit(pp.clamp(1e-4, 1 - 1e-4)) + b * torch.logit(res["pi"].clamp(1e-4, 1 - 1e-4))
        return torch.where(matched, torch.sigmoid(lg), res["pi"])

    def forward(self, x, gt=None, share=0.0, prior=None):
        c = self.cfg
        feats, maps, seed_lg, dirh, raw_dir = self.encode(x)
        B = x.shape[0]
        dev = x.device
        self.pat.to(dev)
        jl = self.junc_head(feats[0][:, :64]) if c.junc else None
        jprob = torch.sigmoid(jl.float()).detach() if (c.junc and c.fork_relax) else None
        sd = self.seeds(maps, raw_dir)
        sel_src = self.add_prior(sd, prior) if prior is not None else sd
        idx, sok = self.seeds_select(sel_src, c.K0)
        s_pos, s_th, s_h, s_val = gather1(sel_src["pos"], idx), gather1(sel_src["th"], idx), gather1(sel_src["h"], idx), gather1(sel_src["val"], idx)
        s_pos, s_th, s_h = grad_scale(s_pos, c.seed_gscale), grad_scale(s_th, c.seed_gscale), grad_scale(s_h, c.seed_gscale)  # S stabilise: the tree losses reach the seed heads with a reduced gradient
        K0 = idx.shape[1]
        # two tips per seed: forward (flag +1) and backward (flag -1, growth against the travel direction)
        pos = torch.cat([s_pos, s_pos], 1)
        th = torch.cat([s_th, s_th + math.pi], 1)
        flag = torch.cat([torch.ones(B, K0, device=dev), -torch.ones(B, K0, device=dev)], 1)
        h = torch.cat([s_h, s_h], 1) + self.flag_emb((flag < 0).long()).to(s_h.dtype)
        valid = torch.cat([sok, sok], 1).float()
        pi0 = torch.sigmoid(torch.cat([s_val, s_val], 1))
        N0 = pos.shape[1]
        nodes = dict(pos=[pos], th_travel=[torch.where(flag > 0, th, th - math.pi)], flag=[flag], nu=[pi0], pi=[pi0], valid=[valid], parent=[torch.full((B, N0), -1, dtype=torch.long, device=dev)], level=[torch.zeros(B, N0, device=dev)], h=[h], root=[torch.arange(K0, device=dev)[None].expand(B, -1).repeat(1, 2)])
        st = dict(pos=pos, th=th, h=h, valid=valid, flag=flag, pi=pi0, gidx=torch.arange(N0, device=dev)[None].expand(B, -1), path=pos[:, :, None, :], pmask=torch.zeros(B, N0, 1, device=dev))
        levels, n_nodes = [], N0
        for l in range(c.D):
            kv = self.kv_set(nodes, st, c)
            relax = self.fork_relax_mask(jprob, st["pos"].detach(), st["th"].detach(), st["flag"]) if jprob is not None else None
            h_ctx, o, th_c, hc = self.level(maps, st, kv, share, gt, relax=relax)
            b = o["end"].shape[2]
            Nn = st["pos"].shape[1]
            end, nu, ok = o["end"], o["nu"], o["ok"]
            pi_par = st["pi"][:, :, None].expand(B, Nn, b)
            pi_c = torch.minimum(pi_par, nu) if c.pi_mode == "min" else pi_par * nu
            fl = st["flag"][:, :, None].expand(B, Nn, b)
            par = st["gidx"][:, :, None].expand(B, Nn, b)
            vflat = (ok & (st["valid"][:, :, None] > 0)).reshape(B, -1)
            pflat = pi_c.reshape(B, -1)
            M = min(c.M, pflat.shape[1])
            flf = fl.reshape(B, -1)  # SELECTION: one beam per growth direction (an untrained or one-sided scorer must not drop a whole direction)
            selF, okF = PG.beam_topm(pflat.detach(), vflat & (flf > 0), M // 2)
            selB, okB = PG.beam_topm(pflat.detach(), vflat & (flf < 0), M - M // 2)
            sel, sok2 = torch.cat([selF, selB], 1), torch.cat([okF, okB], 1)
            n_pos, n_th, n_nu, n_pi, n_fl, n_par = gflat(end, sel), gflat(th_c, sel), gflat(nu, sel), gflat(pi_c, sel), gflat(fl, sel), gflat(par, sel)
            n_h = gflat(hc, sel)
            n_valid = sok2.float()
            # node record (the predicted position stays in the record even if the propagated state is replaced by an expert state)
            th_trav = torch.where(n_fl > 0, n_th, n_th - math.pi)
            nodes["pos"].append(n_pos), nodes["th_travel"].append(th_trav), nodes["flag"].append(n_fl), nodes["nu"].append(n_nu), nodes["pi"].append(n_pi), nodes["valid"].append(n_valid), nodes["parent"].append(n_par), nodes["level"].append(torch.full((B, M), float(l + 1), device=dev)), nodes["h"].append(n_h), nodes["root"].append(torch.cat(nodes["root"], 1).gather(1, n_par))
            levels.append(dict(out=o, tips_gidx=st["gidx"], tips_pos=st["pos"], tips_th=st["th"], tips_flag=st["flag"], tips_valid=st["valid"], tips_pi=st["pi"], sel=sel, sel_ok=sok2, child_pos=end, child_nu=nu, child_ok=ok, n_off=n_nodes))
            # next tips
            pos2, th2 = (n_pos.detach(), n_th.detach()) if c.detach_pos else (n_pos, n_th)
            if share > 0 and gt is not None:
                pos2, th2 = self.expert_state(n_pos, n_th, n_fl, gt, share)
            par_tip = sel // b
            ppos = gather1(st["pos"], par_tip).detach()
            path = torch.cat([gather1(st["path"], par_tip), ppos[:, :, None, :]], 2)
            pmask = torch.cat([gather1(st["pmask"], par_tip), torch.ones(B, M, 1, device=dev)], 2)
            st = dict(pos=pos2, th=th2, h=n_h, valid=n_valid, flag=n_fl, pi=n_pi, gidx=n_nodes + torch.arange(M, device=dev)[None].expand(B, -1), path=path, pmask=pmask)
            n_nodes += M
        res = {k: torch.cat(v, 1) for k, v in nodes.items()}
        res.update(levels=levels, seeds=sd, seed_idx=idx, seed_ok=sok, seed_logit=seed_lg, dirh=dirh, feats=feats, maps=maps)
        if c.junc:
            res["junc_logit"] = jl
        if prior is not None:  # T.2 time recursion: fuse the validity with the warped previous frame (before the keep gate)
            res["pi_single"] = res["pi"]
            res["pi"] = self.fuse_prior(res, prior)
        if c.spf2:  # SPF2: learned keep gate per node (hard concrete, expected-L0 price in the loss); the gated path validity is what the output and the losses use
            kl = self.keep(res["h"].to(self.keep.weight.dtype))[..., 0].float()
            z, p_open = hard_concrete(kl, self.training)
            res.update(keep_logit=kl, z_keep=z, p_open=p_open, pi_tree=res["pi"])
            res["pi"] = res["pi"] * z
        if c.revisit:
            res["pos_rev"], res["keep_rev"] = self.revisit(res, feats)
        if c.constr in ("L", "H"):
            res.update(self.link_head(res, maps, res["pos"].shape[1] - nodes["pos"][-1].shape[1]))
        return res

    def link_head(self, res, maps, off, k=6):
        """L: links between the forward ends (flag +1) and the backward ends (flag -1) of pieces of DIFFERENT seeds. Candidates = the k nearest backward ends inside the cone of the pattern envelope
        (selection from positions); logit = MLP(tokens, relative pose, survival of the straight connector verified at 3 interior points). Returns link_i [B,M], link_j [B,M,k] (node indices in the level slice), logit, mask."""
        pat = self.pat
        pos = res["pos"][:, off:]
        th = res["th_travel"][:, off:]
        flag, h, pi, valid = res["flag"][:, off:], res["h"][:, off:], res["pi"][:, off:], res["valid"][:, off:]
        parent = res["parent"][:, off:]
        K0 = off // 2
        seed_of = res["root"][:, off:]  # the seed (root of the tree) a node grows from: no links inside one tree
        B, M, _ = pos.shape
        d = torch.cdist(pos.detach().float(), pos.detach().float())  # [B,M,M]
        rel = pos[:, None, :, :] - pos[:, :, None, :]  # j - i
        c, s = torch.cos(th), torch.sin(th)
        u = rel[..., 0] * c[:, :, None] + rel[..., 1] * s[:, :, None]
        v = -rel[..., 0] * s[:, :, None] + rel[..., 1] * c[:, :, None]
        phi = torch.atan2(v, u).abs() * 180.0 / math.pi
        reach = 24.0 if pat.kind == "teardrop" else 8.0
        env = torch.clamp(24.0 / (1.0 + (phi / 20.0) ** 2), min=4.0).clamp(max=reach) if pat.kind == "teardrop" else torch.full_like(phi, 8.0)
        cone = (d <= env) & (phi <= (135.0 if pat.kind == "teardrop" else 45.0)) & (u > 0)  # the pattern envelope as a feature (inside the cone), not a hard filter: overlapping pieces must be seen to be rejected
        okp = (flag[:, :, None] > 0) & (flag[:, None, :] < 0) & (seed_of[:, :, None] != seed_of[:, None, :]) & (valid[:, :, None] > 0) & (valid[:, None, :] > 0) & (d <= reach)
        dd = d.masked_fill(~okp, 1e9)
        kd, ki = dd.topk(k, dim=-1, largest=False)  # SELECTION: candidate list from positions
        mask = kd < 1e8
        pj = gather1(pos, ki.reshape(B, -1)).reshape(B, M, k, 2)
        hj = gather1(h, ki.reshape(B, -1)).reshape(B, M, k, -1)
        thj = gather1(th, ki.reshape(B, -1)).reshape(B, M, k)
        pij = gather1(pi, ki.reshape(B, -1)).reshape(B, M, k)
        fr = const([0.25, 0.5, 0.75], pos.device)[None, None, None, :, None]
        qc = pos[:, :, None, None, :] + fr * (pj - pos[:, :, None, :])[:, :, :, None, :]  # connector samples [B,M,k,3,2]
        sc = self.sample(maps, qc)
        dirv = F.normalize(pj - pos[:, :, None, :], dim=-1, eps=1e-6)
        cv = torch.sigmoid(self.conn_v(torch.cat([sc, dirv[:, :, :, None, :].expand(-1, -1, -1, 3, -1), torch.zeros_like(sc[..., :2])], -1)).float()[..., 0]).clamp(0.02, 0.99)
        tconn = cv.prod(-1)
        rl = pj - pos[:, :, None, :]
        ru = rl[..., 0] * c[:, :, None] + rl[..., 1] * s[:, :, None]
        rv = -rl[..., 0] * s[:, :, None] + rl[..., 1] * c[:, :, None]
        dth = thj - th[:, :, None]
        incone = gather1(cone.float().reshape(B, M * M, 1), (torch.arange(M, device=pos.device)[None, :, None] * M + ki).reshape(B, -1)).reshape(B, M, k)
        geo = torch.stack([ru / 24.0, rv / 24.0, kd.clamp(max=30.0) / 24.0, torch.cos(dth), torch.sin(dth), tconn, pi[:, :, None].expand(-1, -1, k), pij, incone], -1)
        f = torch.cat([h[:, :, None, :].expand(-1, -1, k, -1), hj, geo.to(h.dtype)], -1)
        logit = self.link(f)[..., 0].float()
        if self.cfg.link_mode == "geo":  # baseline B1 (K30): TopoLogic-style geometric prior on the end-to-start distance and the heading change, nothing learned
            a = torch.exp(-(kd.clamp(max=60.0) ** 2) / 11.5) * torch.exp(-((torch.atan2(torch.sin(dth), torch.cos(dth)) / 0.6) ** 2))
            logit = torch.logit(a.clamp(1e-4, 1 - 1e-4)) + 0.0 * logit
        return dict(link_ki=ki, link_logit=logit, link_mask=mask, link_off=off, link_tconn=tconn, link_cv=cv)

    def kv_set(self, nodes, st, c):
        """keys of the context layer: the current tips, and for the global context all committed nodes as well"""
        if c.ctx == "global":
            hh = torch.cat([st["h"]] + [t.to(st["h"].dtype) for t in nodes["h"]], 1)
            pp = torch.cat([st["pos"]] + nodes["pos"], 1)
            vv = torch.cat([st["valid"]] + nodes["valid"], 1)
            return dict(h=hh, pos=pp, valid=vv)
        return dict(h=st["h"], pos=st["pos"], valid=st["valid"])

    @staticmethod
    def expert_state(pos, th, flag, gt, share):
        """warm start: a random share of the propagated tips is replaced by the nearest ground-truth point (within 2.5 m) with the travel tangent (stop-gradient state)"""
        with torch.no_grad():
            g05, t05, m05 = gt["g05"], gt["t05"], gt["m05"]
            d = torch.cdist(pos.detach(), g05)
            d = d.masked_fill(m05[:, None, :] < 0.5, 1e9)
            dm, ix = d.min(-1)
            gp = g05.gather(1, ix[..., None].expand(-1, -1, 2))
            gt_th = torch.atan2(t05.gather(1, ix[..., None].expand(-1, -1, 2))[..., 1], t05.gather(1, ix[..., None].expand(-1, -1, 2))[..., 0])
            gt_th = torch.where(flag > 0, gt_th, gt_th + math.pi)
            use = (torch.rand_like(dm) < share) & (dm < 2.5)
        return torch.where(use[..., None], gp, pos), torch.where(use, gt_th, th)

    def revisit(self, res, feats):
        import p15_refine as PR

        p = res["pos"]
        vm = (res["pi"] > 0.1).float() * res["valid"]
        par = res["parent"]
        B, Nn = p.shape[:2]
        has = (par >= 0).float() * vm
        edges = torch.stack([par.clamp(min=0), torch.arange(Nn, device=p.device)[None].expand(B, -1)], -1)
        edges = torch.where((res["flag"] > 0)[..., None], edges, edges.flip(-1))
        t = PR.tangents(p, edges, has)
        h = self.rev_in(res["h"].to(self.rev_in.weight.dtype))
        f0, f1 = feats[0].to(p.dtype), feats[1].to(p.dtype)
        kps = []
        for lyr in self.rev:
            h, d2, kp = lyr(h, p, t, f0, f1, vm)
            kps.append(kp)
            nrm = torch.stack([-t[..., 1], t[..., 0]], -1)
            p = p + (d2[..., 0:1] * t + d2[..., 1:2] * nrm) * vm[..., None]
        return p, torch.stack(kps, 0)  # keep logits of every layer [L,B,Nn]

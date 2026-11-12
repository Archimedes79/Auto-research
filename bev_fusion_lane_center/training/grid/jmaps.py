"""The junction maps head (E286, D82): entry / exit points with heading, the junction zone and the painted evidence (stop lines, pedestrian crossings) as pixel maps, read from the pixel stage of the lane head.

    jm = JMaps(c2=144, c1=48, c05=32, d=64, ctx="dil")           # ctx: "none" | "dil" (dilated residual blocks at 2 m) | "attn" (dil + one self-attention layer at 4 m); "+dirx" appended (E319): the heading-guided exchange `DirX`
    out = jm(P)                                                  # P = the dict of `pyramid.PyramidHead` (feat2, feat1, feat05)
        out["j1"]   (B, 8, 100, 100)   1 m logits: zone, conflict core, entry heat, exit heat, entry heading (tx, ty raw), exit heading (tx, ty raw)
        out["j05"]  (B, 2, 200, 200)   0.5 m logits: stop lines, pedestrian crossings
        out["f1"]   (B, d, 100, 100)   the 1 m features of the head (read by `jends.JEnds`)
    loss, info = jmaps_loss(out, batch)                          # batch carries jz, jheat, jdir, jpaint (`qjunction`)
    kp = keypoints(out, thr=0.3)                                 # per frame: entry / exit keypoints (x, y, tx, ty, p) by a 3 x 3 local maximum + soft-argmax; a hard READ-OUT, not on the training path

Everything on the training path is a soft map (BCE + dice, cosine loss for the heading); the keypoint extraction is for scoring and for the delivered output (named in the card).
E319 `DirX` (ctx "dil+dirx", `jmaps_train.py --dirx`): a heading-guided bidirectional exchange on the 1 m features, between the context blocks and the output convolutions. Every cell reads the features AHEAD and BEHIND it along the lane
heading t(x) of the pyramid's own centre-lane field (`P["centre1"]` channels 1, 2) at 2 / 4 / 8 m (bilinear `grid_sample` at x +- s t(x): dense, differentiable in the features and in the heading), a residual 1 x 1 block mixes them (last layer zero-initialised, so the
model starts as the model without it); two rounds, the second one with a heading refined from the features of the first (zero-initialised correction). The flag lives in the ctx string, so every loader that builds `JModel(ctx=args["ctx"])` builds the right model.
E332 (D84, simplify) `+nopaint` / `+nozone` (`jmaps_train.py --no-paint --no-zone`): the head without its auxiliary maps. `nopaint` drops the 0.5 m branch (`m05`, `head05`, 1.48 of the head's 3.26 GMACs): the output has no `j05`, the loss no paint term;
`nozone` keeps the 1 m output layout (zone / core channels exist but are no longer supervised, nothing reads them) and marks `out["skip"] = ("zone",)` so that `jmaps_loss` leaves the zone / core term out. The pair head must then run `--no-evidence` (the evidence along the
curve reads the zone, core and paint maps). Both flags live in the ctx string; the flags off are the head of E307 bit for bit.
Axes: rows = x (right), columns = y (forward), origin -50 m; 1 m cells (100 x 100) and 0.5 m cells (200 x 200). Run the self-test: python jmaps.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from head import block

NCH1 = 8                                                                 # zone, core, entry heat, exit heat, entry heading (2), exit heading (2)


class ResDil(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(d, d, 3, 1, dil, dilation=dil, bias=False), nn.GroupNorm(8, d), nn.ReLU(inplace=True), nn.Conv2d(d, d, 3, 1, 1, bias=False), nn.GroupNorm(8, d))

    def forward(self, x):
        return F.relu(x + self.f(x))


HALF1 = 50.0                                                             # the 1 m maps cover -50 .. 50 m in x and y


def _unit2(v, eps=1e-4):
    return v / v.norm(dim=1, keepdim=True).clamp_min(eps)


def shift_along(f, t, s, half=HALF1):
    """f (B, C, H, W) read at x + s * t(x): t (B, 2, H, W) the unit heading per cell as (component along the rows = x, along the columns = y), s in metres (negative = behind), bilinear, zero outside the map.
    -> (B, C, H, W) float32; differentiable in f and in t"""
    B, C, H, W = f.shape
    dev = f.device
    ci = -half + (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) * (2 * half / H)
    cj = -half + (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) * (2 * half / W)
    px = ci[None, :, None] + s * t[:, 0].float()
    py = cj[None, None, :] + s * t[:, 1].float()
    g = torch.stack([py, px], -1) / half                                  # grid x = column = y, grid y = row = x (as `jends._sample`)
    return F.grid_sample(f.float(), g, mode="bilinear", padding_mode="zeros", align_corners=False)


class DirX(nn.Module):
    """E319: the heading-guided bidirectional exchange along the lane (a smoother in the Kalman sense, dense and soft). Per round every cell gets the features at x + s t(x) (ahead, 'forward') and x - s t(x) (behind, 'backward'),
    s in `steps` metres, and a residual 1 x 1 block mixes them with the cell's own features and the pyramid's centre-lane probability (where the heading field means something): f <- f + mix(cat[f, ahead_s, behind_s, p]).
    The last layer of every mix is zero-initialised (the model starts as the one without the block). Between the rounds the heading is refined by a zero-initialised 1 x 1 correction of the features, t <- unit(t + turn(f)): the second round follows
    the curve the first round saw. The pyramid heading enters detached (the exchange does not move the pyramid's own heading); the sampling positions are differentiable in the refined heading. Information reaches `rounds * max(steps)` m each way."""

    def __init__(self, d=64, steps=(2.0, 4.0, 8.0), rounds=2, hid=32):
        super().__init__()
        self.steps, self.rounds = tuple(float(s) for s in steps), rounds
        nin = (1 + 2 * len(self.steps)) * d + 1
        self.mix = nn.ModuleList()
        for _ in range(rounds):
            last = nn.Conv2d(hid, d, 1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.mix.append(nn.Sequential(nn.Conv2d(nin, hid, 1, bias=False), nn.GroupNorm(8, hid), nn.ReLU(inplace=True), last))
        self.turn = nn.ModuleList()
        for _ in range(rounds - 1):
            c = nn.Conv2d(d, 2, 1)
            nn.init.zeros_(c.weight)
            nn.init.zeros_(c.bias)
            self.turn.append(c)

    def forward(self, f, P):
        c1 = P["centre1"]
        t = _unit2(c1[:, 1:3].detach().float())
        pc = c1[:, :1].detach().float().sigmoid()
        for r in range(self.rounds):
            parts = [f.float()]
            for s in self.steps:
                parts += [shift_along(f, t, s), shift_along(f, t, -s)]
            parts.append(pc)
            f = f + self.mix[r](torch.cat(parts, 1))
            if r < self.rounds - 1:
                t = _unit2(t + self.turn[r](f).float())
        return f


class JMaps(nn.Module):
    def __init__(self, c2=144, c1=48, c05=32, d=64, ctx="dil"):
        super().__init__()
        base, *mods = ctx.split("+")
        assert base in ("none", "dil", "attn") and set(mods) <= {"dirx", "nopaint", "nozone"}, ctx
        ctx = base
        self.ctx_mode, self.d = ctx, d
        self.paint, self.zone = "nopaint" not in mods, "nozone" not in mods                  # E332: the 0.5 m paint branch / the zone + core supervision of the 1 m branch
        self.p2 = nn.Sequential(nn.Conv2d(c2, d, 1, bias=False), nn.GroupNorm(8, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResDil(d, k) for k in (1, 2, 4, 8)]) if ctx != "none" else nn.ModuleList()
        if ctx == "attn":
            self.pos = nn.Parameter(torch.zeros(1, 25 * 25, d))
            nn.init.normal_(self.pos, std=0.02)
            self.attn = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.m1 = block(c1 + d, d)
        self.head1 = nn.Conv2d(d, NCH1, 1)
        if self.paint:
            self.m05 = block(c05 + d, 32)
            self.head05 = nn.Conv2d(32, 2, 1)
        for m in ((self.head1, self.head05) if self.paint else (self.head1,)):
            nn.init.normal_(m.weight, std=0.01)
        with torch.no_grad():
            self.head1.bias.copy_(torch.tensor([-3.0, -3.0, -4.0, -4.0, 0.0, 1.0, 0.0, 1.0]))
            if self.paint:
                self.head05.bias.copy_(torch.tensor([-4.0, -4.0]))
        self.dirx = DirX(d) if "dirx" in mods else None                      # created LAST: the random initialisation of every other tensor is the same with and without the flag

    def forward(self, P):
        up = lambda t, n: F.interpolate(t, scale_factor=n, mode="bilinear", align_corners=False)  # noqa: E731
        c = self.p2(P["feat2"])                                              # (B, d, 50, 50) the 2 m features of the pyramid
        for b in self.blocks:
            c = b(c)
        if self.ctx_mode == "attn":
            t = F.avg_pool2d(c, 2)                                           # 25 x 25 tokens (4 m)
            B, d = t.shape[:2]
            t = self.attn(t.flatten(2).transpose(1, 2) + self.pos)
            c = c + up(t.transpose(1, 2).reshape(B, d, 25, 25), 2)[:, :, :50, :50]
        x1 = self.m1(torch.cat([P["feat1"], up(c, 2)], 1))
        if self.dirx is not None:                                            # E319: forward / backward exchange along the lane heading, before the output convolutions and before `f1` is read by the pair head
            x1 = self.dirx(x1, P)
        j1 = self.head1(x1)
        if not self.paint:                                                   # E332: no 0.5 m branch; `skip` names the loss terms that are left out
            return dict(j1=j1, f1=x1) if self.zone else dict(j1=j1, f1=x1, skip=("zone",))
        j05 = self.head05(self.m05(torch.cat([P["feat05"], up(x1, 2)], 1)))
        return dict(j1=j1, j05=j05, f1=x1) if self.zone else dict(j1=j1, j05=j05, f1=x1, skip=("zone",))


class JModel(nn.Module):
    """the pixel stage of the lane head (pyramid) + the junction maps head; `pyr` is loaded from a baseline run (`runs/e273_p_s<k>/model.pt`, tensors `pyr.*`), so the junction maps read the SAME pixel features as B"""

    def __init__(self, width=48, f05=32, d=64, ctx="dil"):
        super().__init__()
        from pyramid import PyramidHead
        self.width, self.ctx = width, ctx
        self.pyr = PyramidHead(width=width, f05=f05)
        self.jm = JMaps(c2=3 * width, c1=width, c05=f05, d=d, ctx=ctx)

    def load_base(self, path):
        sd = torch.load(path, map_location="cpu")["state"]
        own = self.pyr.state_dict()
        fit = {k[4:]: v for k, v in sd.items() if k.startswith("pyr.") and k[4:] in own and own[k[4:]].shape == v.shape}
        self.pyr.load_state_dict(fit, strict=False)
        return len(fit), len(own)

    def forward(self, x, joint=False):
        if joint:
            P = self.pyr(x)
        else:
            with torch.no_grad():
                P = self.pyr(x)
        return P, self.jm(P)


def _dice(logit, y, eps=1.0):
    p = logit.sigmoid()
    inter = (p * y).sum((-2, -1))
    return (1 - (2 * inter + eps) / (p.sum((-2, -1)) + y.sum((-2, -1)) + eps)).mean()


def jmaps_loss(out, batch, far_w=0.0, far_r=35.0):
    dev = out["j1"].device
    j1 = out["j1"].float()
    j05 = out["j05"].float() if "j05" in out else None                                                  # E332: a head without the paint branch has no j05 and no paint loss
    jz, jh, jd, jp = (batch[k].to(dev).float() for k in ("jz", "jheat", "jdir", "jpaint"))
    info = {}
    if "zone" in out.get("skip", ()):                                                                   # E332: the zone / core maps are not supervised (nothing reads them)
        l_z = 0.0
    else:
        l_z = F.binary_cross_entropy_with_logits(j1[:, :2], jz) + _dice(j1[:, :2], jz)
    info["zone"] = float(l_z)
    pw = torch.tensor(20.0, device=dev)
    if far_w > 0:                                                                                      # E304: the cells beyond far_r metres from the ego count 1 + far_w times in the heat BCE (53 % of the GT ends are there, the recall is 0.55)
        cc = -50.0 + (torch.arange(100, device=dev).float() + 0.5)
        wmap = 1.0 + far_w * ((cc[:, None] ** 2 + cc[None, :] ** 2).sqrt() > far_r).float()
        l_h = (F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw, reduction="none") * wmap).mean() + _dice(j1[:, 2:4], jh)
    else:
        l_h = F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw) + _dice(j1[:, 2:4], jh)
    info["heat"] = float(l_h)
    l_d = 0.0
    for k in range(2):                                                       # heading where the target has one: cosine loss on the normalised raw output
        tgt = jd[:, 2 * k: 2 * k + 2]
        m = (tgt.abs().sum(1) > 0.5).float()
        pr = j1[:, 4 + 2 * k: 6 + 2 * k]
        pr = pr / pr.norm(dim=1, keepdim=True).clamp_min(1e-4)
        l_d = l_d + ((1 - (pr * tgt).sum(1)) * m).sum() / m.sum().clamp_min(1.0)
    info["dir"] = float(l_d)
    l_p = F.binary_cross_entropy_with_logits(j05, jp, pos_weight=torch.tensor(10.0, device=dev)) + _dice(j05, jp) if j05 is not None else 0.0
    info["paint"] = float(l_p)
    return l_z + l_h + 0.5 * l_d + l_p, info


@torch.no_grad()
def keypoints(out, thr=0.3, half=50.0, res=1.0):
    """entry / exit keypoints per frame: 3 x 3 local maxima of the heat above thr, position by the soft-argmax of the 3 x 3 window (sub-cell), heading from the heading channels at the peak.
    -> list over frames of dict(ent (n, 5) x, y, tx, ty, p ; ext (m, 5))"""
    heat = out["j1"][:, 2:4].float().sigmoid()
    B, _, H, W = heat.shape
    mx = F.max_pool2d(heat, 3, 1, 1)
    pk = (heat >= mx) & (heat > thr)
    ii = (torch.arange(H, device=heat.device).float() + 0.5) * res - half
    res_out = []
    for b in range(B):
        d = {}
        for k, name in enumerate(("ent", "ext")):
            idx = pk[b, k].nonzero()
            rows = []
            for i, j in idx.tolist():
                i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, H), max(j - 1, 0), min(j + 2, W)
                w = heat[b, k, i0:i1, j0:j1]
                wsum = w.sum()
                x = (w.sum(1) * ii[i0:i1]).sum() / wsum
                y = (w.sum(0) * ii[j0:j1]).sum() / wsum
                t = out["j1"][b, 4 + 2 * k: 6 + 2 * k, i, j].float()
                t = t / t.norm().clamp_min(1e-4)
                rows.append([float(x), float(y), float(t[0]), float(t[1]), float(heat[b, k, i, j])])
            d[name] = torch.tensor(rows).reshape(-1, 5)
        res_out.append(d)
    return res_out


def _selftest():
    torch.manual_seed(0)
    P = dict(feat2=torch.randn(2, 144, 50, 50), feat1=torch.randn(2, 48, 100, 100), feat05=torch.randn(2, 32, 200, 200), centre1=torch.randn(2, 12, 100, 100))
    for ctx in ("none", "dil", "attn", "dil+dirx", "dil+nopaint", "dil+nopaint+nozone", "dil+dirx+nopaint+nozone"):
        m = JMaps(ctx=ctx)
        out = m(P)
        assert out["j1"].shape == (2, 8, 100, 100) and (("j05" in out and out["j05"].shape == (2, 2, 200, 200)) == ("nopaint" not in ctx)), (ctx, out["j1"].shape)
        assert ("skip" in out) == ("nozone" in ctx), ctx
        batch = dict(jz=torch.zeros(2, 2, 100, 100), jheat=torch.zeros(2, 2, 100, 100), jdir=torch.zeros(2, 4, 100, 100), jpaint=torch.zeros(2, 2, 200, 200))
        batch["jz"][:, :, 40:60, 40:60] = 1
        batch["jheat"][:, 0, 30, 50] = 1
        batch["jdir"][:, 0, 30, 50], batch["jdir"][:, 1, 30, 50] = 0.0, 1.0
        loss, info = jmaps_loss(out, batch)
        lf, _ = jmaps_loss(out, batch, far_w=2.0)                                                         # E304: the far weight changes the loss and nothing else
        assert abs(float(lf) - float(loss)) > 1e-6
        loss.backward()
        if m.dirx is not None:                                                                            # zero-initialised last layers: the first step opens the gradient of everything before them
            for _ in range(2):
                with torch.no_grad():
                    for p in m.parameters():
                        if p.grad is not None:
                            p.sub_(0.05 * p.grad)
                m.zero_grad()
                loss, _ = jmaps_loss(m(P), batch)
                loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        assert not dead, (ctx, dead)
    # keypoints: a clean peak at cell (30, 50) is found at x = -19.5, y = 0.5
    o = dict(j1=torch.full((1, 8, 100, 100), -6.0))
    o["j1"][0, 2, 30, 50] = 6.0
    o["j1"][0, 5, 30, 50] = 5.0
    o["j1"][0, 4, 30, 50] = 0.0
    kp = keypoints(o, thr=0.5)[0]
    assert len(kp["ent"]) == 1 and len(kp["ext"]) == 0 and abs(kp["ent"][0, 0] + 19.5) < 0.1 and abs(kp["ent"][0, 1] - 0.5) < 0.1 and kp["ent"][0, 3] > 0.99, kp
    print("jmaps selftest ok; params", sum(p.numel() for p in JMaps(ctx="attn").parameters()) / 1e6, "M")


if __name__ == "__main__":
    _selftest()

"""The junction maps head (E286, D82): entry / exit points with heading, the junction zone and the painted evidence (stop lines, pedestrian crossings) as pixel maps, read from the pixel stage of the lane head.

    jm = JMaps(c2=144, c1=48, c05=32, d=64, ctx="dil")           # ctx: "none" | "dil" (dilated residual blocks at 2 m) | "attn" (dil + one self-attention layer at 4 m)
    out = jm(P)                                                  # P = the dict of `pyramid.PyramidHead` (feat2, feat1, feat05)
        out["j1"]   (B, 8, 100, 100)   1 m logits: zone, conflict core, entry heat, exit heat, entry heading (tx, ty raw), exit heading (tx, ty raw)
        out["j05"]  (B, 2, 200, 200)   0.5 m logits: stop lines, pedestrian crossings
        out["f1"]   (B, d, 100, 100)   the 1 m features of the head (read by `jends.JEnds`)
    loss, info = jmaps_loss(out, batch)                          # batch carries jz, jheat, jdir, jpaint (`qjunction`)
    kp = keypoints(out, thr=0.3)                                 # per frame: entry / exit keypoints (x, y, tx, ty, p) by a 3 x 3 local maximum + soft-argmax; a hard READ-OUT, not on the training path

Everything on the training path is a soft map (BCE + dice, cosine loss for the heading); the keypoint extraction is for scoring and for the delivered output (named in the card).
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


class JMaps(nn.Module):
    def __init__(self, c2=144, c1=48, c05=32, d=64, ctx="dil"):
        super().__init__()
        assert ctx in ("none", "dil", "attn"), ctx
        self.ctx_mode, self.d = ctx, d
        self.p2 = nn.Sequential(nn.Conv2d(c2, d, 1, bias=False), nn.GroupNorm(8, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResDil(d, k) for k in (1, 2, 4, 8)]) if ctx != "none" else nn.ModuleList()
        if ctx == "attn":
            self.pos = nn.Parameter(torch.zeros(1, 25 * 25, d))
            nn.init.normal_(self.pos, std=0.02)
            self.attn = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.m1 = block(c1 + d, d)
        self.head1 = nn.Conv2d(d, NCH1, 1)
        self.m05 = block(c05 + d, 32)
        self.head05 = nn.Conv2d(32, 2, 1)
        for m in (self.head1, self.head05):
            nn.init.normal_(m.weight, std=0.01)
        with torch.no_grad():
            self.head1.bias.copy_(torch.tensor([-3.0, -3.0, -4.0, -4.0, 0.0, 1.0, 0.0, 1.0]))
            self.head05.bias.copy_(torch.tensor([-4.0, -4.0]))

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
        j1 = self.head1(x1)
        j05 = self.head05(self.m05(torch.cat([P["feat05"], up(x1, 2)], 1)))
        return dict(j1=j1, j05=j05, f1=x1)


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


def jmaps_loss(out, batch):
    dev = out["j1"].device
    j1, j05 = out["j1"].float(), out["j05"].float()
    jz, jh, jd, jp = (batch[k].to(dev).float() for k in ("jz", "jheat", "jdir", "jpaint"))
    info = {}
    l_z = F.binary_cross_entropy_with_logits(j1[:, :2], jz) + _dice(j1[:, :2], jz)
    info["zone"] = float(l_z)
    pw = torch.tensor(20.0, device=dev)
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
    l_p = F.binary_cross_entropy_with_logits(j05, jp, pos_weight=torch.tensor(10.0, device=dev)) + _dice(j05, jp)
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
    P = dict(feat2=torch.randn(2, 144, 50, 50), feat1=torch.randn(2, 48, 100, 100), feat05=torch.randn(2, 32, 200, 200))
    for ctx in ("none", "dil", "attn"):
        m = JMaps(ctx=ctx)
        out = m(P)
        assert out["j1"].shape == (2, 8, 100, 100) and out["j05"].shape == (2, 2, 200, 200), (ctx, out["j1"].shape)
        batch = dict(jz=torch.zeros(2, 2, 100, 100), jheat=torch.zeros(2, 2, 100, 100), jdir=torch.zeros(2, 4, 100, 100), jpaint=torch.zeros(2, 2, 200, 200))
        batch["jz"][:, :, 40:60, 40:60] = 1
        batch["jheat"][:, 0, 30, 50] = 1
        batch["jdir"][:, 0, 30, 50], batch["jdir"][:, 1, 30, 50] = 0.0, 1.0
        loss, info = jmaps_loss(out, batch)
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

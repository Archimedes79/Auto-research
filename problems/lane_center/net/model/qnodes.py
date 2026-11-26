"""Stage B of the lane head (E273): one node slot per 2 m cell, with existence, type flags, an offset that pulls the node onto the line, a direction and the section / long-link scores.

    NodeHead()(feat2, feat1, feat05, road2, centre1) -> dict
        e (B, 50, 50) existence logit; flags (B, 8, 50, 50) logits in the order of `qtargets.FLAGS`; off (B, 2, 50, 50) offset in metres inside the cell (tanh, +-1 m); dir_raw (B, 2, 50, 50)
        (normalise with `unit`); s (B, 50, 50) section-start logit; uout, uin (B, 50, 50) logits: the node is the start / the end of a LONG link; tok (B, d, 50, 50) the node token; pos (B, 2, 50, 50)
        the node position in metres (cell centre + off), rows = x (right), columns = y (forward)
The token of a cell = the 2 m feature + the pooled 1 m / 0.5 m features + road and intersection probabilities + the pooled centre-lane and event probabilities of the pixel stage (8 channels) + a Fourier code of the cell centre; two layers of neighbourhood attention
(7 x 7 cells = +-6 m, a relative-position bias; the card says 9 x 9: smaller for memory, `k=9` is a parameter) give it the context (a split is only a split if the lane diverges ahead);
then the REFINEMENT: the 1 m and 0.5 m feature maps are sampled bilinearly at the pulled position (differentiable), the token is updated and offset, direction and flags are re-read.
Run the self-test: python qnodes.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N, RES, HALF = 50, 2.0, 50.0
NF = 8
CH = dict(e=0, flags=slice(1, 9), off=slice(9, 11), dir=slice(11, 13), s=13, uout=14, uin=15)


def cell_centres(device="cpu", n=N, res=RES):
    c = -HALF + (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * res
    return torch.stack([c.view(n, 1).expand(n, n), c.view(1, n).expand(n, n)], 0)[None]          # (1, 2, n, n): x (row), y (column)


def fourier(c, nf=4):
    k = (2.0 ** torch.arange(nf, device=c.device, dtype=c.dtype)) * math.pi / HALF
    a = c[:, :, None] * k.view(1, 1, nf, 1, 1)                                                    # (1, 2, nf, H, W)
    return torch.cat([a.sin(), a.cos()], 2).flatten(1, 2)                                         # (1, 4 nf, H, W)


def sample(feat, xy):
    """bilinear sample of a map (B, C, H, W) that spans +-50 m (rows = x, columns = y) at xy (B, 2, h, w) metres -> (B, C, h, w); positions outside are clamped to the border"""
    grid = torch.stack([xy[:, 1] / HALF, xy[:, 0] / HALF], -1).float()                         # float32 even under bf16 autocast (a bf16 grid has 0.2 m steps beyond 25 m)
    return F.grid_sample(feat.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(feat.dtype)


def unit(v, dim=1, eps=1e-6):
    return v / v.norm(dim=dim, keepdim=True).clamp_min(eps)


class LocalAttn(nn.Module):
    """neighbourhood attention on a grid (B, d, H, W): every cell attends to the k x k cells around it; pre-norm, residual, feed-forward"""

    def __init__(self, d, heads=4, k=7):
        super().__init__()
        self.d, self.h, self.k = d, heads, k
        self.n1, self.n2 = nn.GroupNorm(1, d), nn.GroupNorm(1, d)
        self.qkv, self.proj = nn.Conv2d(d, 3 * d, 1), nn.Conv2d(d, d, 1)
        self.ffn = nn.Sequential(nn.Conv2d(d, 2 * d, 1), nn.GELU(), nn.Conv2d(2 * d, d, 1))
        self.bias = nn.Parameter(torch.zeros(heads, k * k))

    def forward(self, x):
        B, d, H, W = x.shape
        h, k, dh = self.h, self.k, self.d // self.h
        q, kk, v = self.qkv(self.n1(x)).chunk(3, 1)
        unf = lambda t: F.unfold(t, k, padding=k // 2).view(B, h, dh, k * k, H * W)  # noqa: E731
        valid = F.unfold(torch.ones(1, 1, H, W, device=x.device), k, padding=k // 2).view(1, 1, k * k, H * W) > 0.5
        att = (q.view(B, h, dh, 1, H * W) * unf(kk)).sum(2) / math.sqrt(dh) + self.bias.view(1, h, k * k, 1)
        att = att.masked_fill(~valid, float("-inf")).softmax(2)                                    # (B, h, k*k, HW)
        out = (att.unsqueeze(2) * unf(v)).sum(3).reshape(B, d, H, W)
        x = x + self.proj(out)
        return x + self.ffn(self.n2(x))


class NodeHead(nn.Module):
    def __init__(self, c2=144, c1=48, c05=32, d=64, heads=4, layers=2, k=7):
        super().__init__()
        self.d = d
        self.inp = nn.Sequential(nn.Conv2d(c2 + c1 + c05 + 2 + 16 + 8, d, 1), nn.GroupNorm(8, d), nn.GELU())
        self.att = nn.ModuleList([LocalAttn(d, heads, k) for _ in range(layers)])
        self.head1 = nn.Conv2d(d, 16, 1)
        self.fine = nn.Sequential(nn.Conv2d(d + c1 + c05 + 4, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.head2 = nn.Conv2d(d, 16, 1)
        nn.init.normal_(self.head1.weight, std=0.01)
        nn.init.normal_(self.head2.weight, std=0.001)
        with torch.no_grad():
            b = torch.zeros(16)
            b[CH["e"]] = -4.0
            b[CH["flags"]] = -4.0
            b[11], b[12] = 0.0, 1.0                                                                # direction forward
            b[CH["s"]], b[CH["uout"]], b[CH["uin"]] = -2.0, -3.0, -3.0
            self.head1.bias.copy_(b)
            self.head2.bias.zero_()

    def forward(self, feat2, feat1, feat05, road2, centre1):
        B = feat2.shape[0]
        c = cell_centres(feat2.device).expand(B, -1, -1, -1)
        cen, ev = centre1[:, :1].float().sigmoid(), centre1[:, 6:12].float().sigmoid()           # the pixel stage's own centre-lane and event probabilities, pooled to the 2 m grid
        pm = torch.cat([F.avg_pool2d(cen, 2), F.max_pool2d(cen, 2), F.avg_pool2d(ev, 2)], 1)
        t = self.inp(torch.cat([feat2, F.avg_pool2d(feat1, 2), F.avg_pool2d(feat05, 4), road2.sigmoid(), fourier(c[:1]).expand(B, -1, -1, -1), pm.to(feat2.dtype)], 1))
        for a in self.att:
            t = a(t)
        o1 = self.head1(t)
        off1 = torch.tanh(o1[:, CH["off"]])
        p1 = c + off1
        t2 = t + self.fine(torch.cat([t, sample(feat1, p1), sample(feat05, p1), off1, unit(o1[:, CH["dir"]])], 1))
        o = o1 + self.head2(t2)
        off = torch.tanh(o[:, CH["off"]])
        return dict(e=o[:, CH["e"]], flags=o[:, CH["flags"]], off=off, dir_raw=o[:, CH["dir"]], s=o[:, CH["s"]], uout=o[:, CH["uout"]], uin=o[:, CH["uin"]], tok=t2, pos=c + off)


def soft_suppress_logit(e_logit, pos, radius=1.5, tau=0.1, iters=3):
    """E277: the close-node suppression of E275 (`evaluation/qgraph_readout.suppress_close_nodes`) as a SOFT step of the forward path: every cell is damped by the cells around it that are more
    probable and closer than `radius` metres, e'_i = e_i * prod_j (1 - e'_j * k_ij) over the 8 neighbour cells j with e_j > e_i (a tie: the lower cell index is the stronger), with the smooth radius
    k_ij = sigmoid((radius - d_ij) / tau) of the predicted node positions. The order (who is stronger) is a comparison without gradient, exactly the order of the hard rule; the neighbour's damped value e'_j
    is used from the second pass on, so a chain A > B > C (B within `radius` of A, C of B, C not of A) keeps C like the hard rule does (greedy order): `iters` passes are exact for chains of at least
    `iters` nodes (the critic measured one more), beyond that the errors alternate between over- and under-damping (the 2 m grid and radius 1.5 m make longer chains rare). At the production setting
    (tau 0.1, 3 passes) the kept set differs from the hard rule in about 1 % of the cells for saturated probabilities (0.85 - 0.99) and in about 7 % for probabilities of 0.5 - 0.9: the step is soft. Only the 3 x 3 neighbourhood matters: two nodes in cells two apart are at least 4 - 2 = 2 m apart (offsets are +-1 m), so for radius <= 2 m
    nothing is missed. In the limit tau -> 0 with e in {0, 1}-like values it returns the kept set of the hard rule (tested in test_soft_nms.py); at tau > 0 the gradient reaches the stronger node's
    probability, the weaker one's probability and both positions. e_logit (B, H, W), pos (B, 2, H, W) metres -> logit of e' (B, H, W), float32."""
    assert iters >= 1 and iters == int(iters) and tau > 0.0 and 0.0 < radius <= 2.0, (radius, tau, iters)
    iters = int(iters)               # tau = 0 is a NaN; a radius above 2 m would miss pairs two cells apart (offsets are +-1 m)
    e = e_logit.float().sigmoid()
    pos = pos.float()
    B, H, W = e.shape
    idx = torch.arange(H * W, device=e.device).view(1, H, W)
    ep = F.pad(e[:, None], (1, 1, 1, 1), value=0.0)[:, 0]                                      # a missing neighbour has e 0: no damping
    pp = F.pad(pos, (1, 1, 1, 1), value=0.0)
    ip = F.pad(idx[:, None].float(), (1, 1, 1, 1), value=float(H * W))[:, 0]
    shifts = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2) if (dy, dx) != (1, 1)]
    dom, kk = [], []
    ed = e.detach()
    for dy, dx in shifts:
        ej = ep[:, dy:dy + H, dx:dx + W]
        pj = pp[:, :, dy:dy + H, dx:dx + W]
        ij = ip[:, dy:dy + H, dx:dx + W]
        d = ((pos - pj).square().sum(1) + 1e-8).sqrt()
        kk.append(torch.sigmoid((radius - d) / tau))
        dom.append(((ej.detach() > ed) | ((ej.detach() == ed) & (ij < idx))) & (ij < H * W))
    cur = e
    for _ in range(iters):
        s = torch.ones_like(e)
        for t, (dy, dx) in enumerate(shifts):
            cj = F.pad(cur[:, None], (1, 1, 1, 1), value=0.0)[:, 0, dy:dy + H, dx:dx + W]       # the neighbour's current (damped) value
            s = s * (1.0 - cj * kk[t] * dom[t].float())
        cur = e * s
    # logit(e * s) without a clamp (critic pass 2026-11-08: a clamp at 1e-6 killed the gradient beyond |logit| 13.8): e * s = sigmoid(l) * s and 1 - e * s = sigmoid(-l) + e * (1 - s); for s = 1 this is l itself
    l = e_logit.float()
    return F.logsigmoid(l) + torch.log(s.clamp_min(1e-12)) - torch.log((torch.sigmoid(-l) + e * (1.0 - s)).clamp_min(1e-30))


def _selftest():
    torch.manual_seed(0)
    B = 2
    f2, f1, f05 = (torch.randn(B, 24, 50, 50, requires_grad=True), torch.randn(B, 16, 100, 100, requires_grad=True), torch.randn(B, 8, 200, 200, requires_grad=True))
    road = torch.randn(B, 2, 50, 50)
    cen1 = torch.randn(B, 12, 100, 100)
    m = NodeHead(c2=24, c1=16, c05=8, d=32, heads=4, layers=2, k=7)
    o = m(f2, f1, f05, road, cen1)
    assert o["e"].shape == (B, 50, 50) and o["flags"].shape == (B, 8, 50, 50) and o["off"].shape == (B, 2, 50, 50) and o["tok"].shape == (B, 32, 50, 50) and o["pos"].shape == (B, 2, 50, 50)
    assert float(o["off"].abs().max()) <= 1.0 and torch.allclose(o["pos"][0, :, 0, 0], torch.tensor([-49.0, -49.0]) + o["off"][0, :, 0, 0])
    loss = sum(v.square().mean() for v in o.values())
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead
    assert f1.grad is not None and float(f1.grad.abs().sum()) > 0 and float(f05.grad.abs().sum()) > 0, "the refinement must reach the fine features"
    # sample(): a map equal to the x coordinate is sampled back as x (rows = x), a map equal to y as y
    xs = torch.linspace(-49.75, 49.75, 200)
    mx = xs.view(1, 1, 200, 1).expand(1, 1, 200, 200).contiguous()
    my = xs.view(1, 1, 1, 200).expand(1, 1, 200, 200).contiguous()
    pt = torch.tensor([[[[12.3]], [[-7.1]]]])                                                    # x = 12.3, y = -7.1
    assert abs(float(sample(mx, pt)) - 12.3) < 0.02 and abs(float(sample(my, pt)) + 7.1) < 0.02, (sample(mx, pt), sample(my, pt))
    n = sum(p.numel() for p in NodeHead().parameters())
    print("params", n / 1e6, "M")
    print("qnodes selftest ok")


if __name__ == "__main__":
    _selftest()

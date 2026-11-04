"""Stage C of the lane head (E273): the connection matrix between nodes, with a Hermite curve for every link.

    LinkHead(d)(no, cen, gt=None, K=512, KL=32) -> dict

no   the node outputs of `qnodes.NodeHead` (e, flags, dir_raw, uout, uin, tok, pos); cen (B, 1, 100, 100) the centre-lane probability (the evidence the curves are checked against);
gt   in training the targets `exists` (B, 50, 50), `uout`, `uin`, `lk` (B, L, 4) cells of the GT links (-1 padded): the true nodes and links are always among the candidates.
Candidates (index choices with the gradient through the values, named in the card):
  active nodes   the K cells with the largest existence (and above 0.05, or a GT node in training)
  short links    pairs of active nodes within 4 cells (+-8 m) in each axis, not behind the node (the offset of j in i's frame along > -1 m)
  long links     the top-KL nodes by `uout` x the top-KL nodes by `uin`, 8 m < distance <= 100 m, inside a 75 degree forward cone (through intersections: the stop -> exit connectors)
Link logit = MLP([token i, token j, geometry (offset of j in i's frame along / across / distance at two scales, heading change cos / sin, long flag), evidence along the curve (mean, minimum,
gap share of the centre probability at 16 points of the Hermite curve)]); the curve is a cubic Hermite segment from (pos i, dir i) to (pos j, dir j) with handles a, b x chord: 1/3 for short
links, 0.9 x sigmoid(MLP) for long links (the learned scales; the targets are the fitted scales of `qtargets.fit_long_scales`).
Output (dict): act (B, K) flat cell indices, valid (B, K), pos (B, K, 2), dirv (B, K, 2), pb / pa / pc (P,) batch element, source and target index in the active list, is_long (P,), logit (P,),
sa / sb (P,) handle scales, label (P,) float (when gt is given), cell (B, K, 2).
Run the self-test: python qlinks.py
"""
import math

import torch
import torch.nn as nn

from qnodes import N, sample, unit

SHORT = 8.0
CONE = math.cos(math.radians(75.0))
NPT = 16


def hermite_t(p0, h0, p1, h1, a, b, n=NPT):
    """cubic Hermite curves (P, n, 2) from (p0, h0) to (p1, h1), handles a, b x chord (all (P,) / (P, 2))"""
    c = (p1 - p0).norm(dim=-1, keepdim=True)
    s = torch.linspace(0, 1, n, device=p0.device, dtype=p0.dtype).view(1, n, 1)
    q1, q2 = p0 + a[:, None] * c * h0, p1 - b[:, None] * c * h1
    p0, p1, q1, q2 = p0[:, None], p1[:, None], q1[:, None], q2[:, None]
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * p1


def gather(m, idx):
    """m (B, C, N, N) maps, idx (B, K) flat cell indices -> (B, K, C)"""
    B, C = m.shape[:2]
    return m.flatten(2).gather(2, idx[:, None, :].expand(B, C, -1)).transpose(1, 2)


class LinkHead(nn.Module):
    def __init__(self, d=64, hidden=128):
        super().__init__()
        nin = 2 * d + 9
        self.scale = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, 2))
        self.mlp = nn.Sequential(nn.Linear(nin + 2 + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.scale[-1].weight, std=0.01)
        with torch.no_grad():
            self.scale[-1].bias.fill_(math.log(0.333 / 0.9 / (1 - 0.333 / 0.9)))              # sigmoid(.) * 0.9 = 1/3 at the start
            self.mlp[-1].bias.fill_(-4.0)

    def forward(self, no, cen, gt=None, K=512, KL=32, thr=0.05):
        B = no["e"].shape[0]
        dev = no["e"].device
        ep = no["e"].detach().sigmoid().flatten(1)
        score = ep + (2.0 * gt["exists"].flatten(1) if gt is not None else 0.0)
        K = min(K, ep.shape[1])
        sv, idx = score.topk(K, dim=1)
        valid = sv > thr
        ci, cj = idx // N, idx % N
        pos = gather(no["pos"], idx)
        dirv = unit(gather(no["dir_raw"], idx), dim=-1)
        tok = gather(no["tok"], idx)
        # candidate masks
        eye = torch.eye(K, dtype=torch.bool, device=dev)[None]
        both = valid[:, :, None] & valid[:, None, :] & ~eye
        dp = pos[:, None, :, :] - pos[:, :, None, :]                                           # [b, a, c] = pos_c - pos_a
        along = (dp * dirv[:, :, None, :]).sum(-1)
        dist = dp.norm(dim=-1)
        near = ((ci[:, :, None] - ci[:, None, :]).abs() <= 4) & ((cj[:, :, None] - cj[:, None, :]).abs() <= 4)
        short = both & near & (along > -1.0)
        us = no["uout"].detach().sigmoid().flatten(1)
        ui = no["uin"].detach().sigmoid().flatten(1)
        if gt is not None:
            us, ui = us + 2.0 * gt["uout"].flatten(1), ui + 2.0 * gt["uin"].flatten(1)
        us_a, ui_a = us.gather(1, idx), ui.gather(1, idx)
        ent = torch.zeros_like(valid).scatter_(1, us_a.topk(min(KL, K), dim=1)[1], True) & valid
        ext = torch.zeros_like(valid).scatter_(1, ui_a.topk(min(KL, K), dim=1)[1], True) & valid
        long = both & ent[:, :, None] & ext[:, None, :] & (dist > SHORT) & (dist <= 100.0) & (along / dist.clamp_min(1e-6) > CONE)
        mask = short | long
        fa = idx                                                                               # flat cell of each active node
        pair_key = None
        if gt is not None:
            lk = gt["lk"]
            okl = lk[..., 0] >= 0
            ga = (lk[..., 0].clamp_min(0) * N + lk[..., 1].clamp_min(0))
            gc = (lk[..., 2].clamp_min(0) * N + lk[..., 3].clamp_min(0))
            inv = torch.full((B, N * N), -1, dtype=torch.long, device=dev)
            inv.scatter_(1, idx, torch.arange(K, device=dev)[None].expand(B, -1))
            ia, ic = inv.gather(1, ga), inv.gather(1, gc)
            okg = okl & (ia >= 0) & (ic >= 0)
            bb = torch.arange(B, device=dev)[:, None].expand_as(ia)
            mask = mask.clone()
            mask[bb[okg], ia[okg], ic[okg]] = True
            pair_key = ((bb[okl] * N * N + ga[okl]) * (N * N) + gc[okl])
        pb, pa, pc = mask.nonzero(as_tuple=True)
        if len(pb) > 80000:
            keep = torch.randperm(len(pb), device=dev)[:80000].sort()[0]
            pb, pa, pc = pb[keep], pa[keep], pc[keep]
        p0, p1, h0, h1 = pos[pb, pa], pos[pb, pc], dirv[pb, pa], dirv[pb, pc]
        d_ = p1 - p0
        dd = d_.norm(dim=-1)
        al = (d_ * h0).sum(-1)
        ac = d_[:, 0] * h0[:, 1] - d_[:, 1] * h0[:, 0]
        is_long = dd > SHORT
        geo = torch.stack([al / 8, ac / 8, dd / 8, al / 50, ac / 50, dd / 50, (h0 * h1).sum(-1), h0[:, 0] * h1[:, 1] - h0[:, 1] * h1[:, 0], is_long.float()], -1)
        x0 = torch.cat([tok[pb, pa], tok[pb, pc], geo], -1)
        sc = 0.9 * torch.sigmoid(self.scale(x0))
        sa = torch.where(is_long, sc[:, 0], torch.full_like(sc[:, 0], 1 / 3))
        sb = torch.where(is_long, sc[:, 1], torch.full_like(sc[:, 1], 1 / 3))
        curve = hermite_t(p0, h0, p1, h1, sa, sb)                                              # (P, NPT, 2)
        ev = torch.zeros(len(pb), NPT, device=dev, dtype=cen.dtype)
        for b in range(B):                                                                     # pairs are sorted by batch element
            sel = (pb == b).nonzero(as_tuple=True)[0]
            if len(sel):
                xy = curve[sel].permute(2, 0, 1)[None]                                         # (1, 2, P_b, NPT)
                ev[sel] = sample(cen[b: b + 1], xy.to(cen.dtype))[0, 0]
        evf = torch.stack([ev.mean(1), ev.min(1)[0], torch.sigmoid((0.2 - ev) * 20).mean(1)], -1)
        logit = self.mlp(torch.cat([x0, sa[:, None], sb[:, None], evf], -1))[:, 0]
        out = dict(act=idx, valid=valid, pos=pos, dirv=dirv, tok=tok, cell=torch.stack([ci, cj], -1), pb=pb, pa=pa, pc=pc, is_long=is_long, logit=logit, sa=sa, sb=sb, evidence=evf)
        if gt is not None:
            key = (pb * N * N + fa[pb, pa]) * (N * N) + fa[pb, pc]
            out["label"] = torch.isin(key, pair_key).float()
            out["gt_missing"] = int(okl.sum()) - int(torch.isin(pair_key, key).sum())          # GT links that did not make it into the candidate list (trimmed or outside K)
        return out


def _selftest():
    torch.manual_seed(0)
    B, d = 2, 16
    no = dict(e=torch.full((B, N, N), -6.0), uout=torch.full((B, N, N), -6.0), uin=torch.full((B, N, N), -6.0), dir_raw=torch.zeros(B, 2, N, N), tok=torch.randn(B, d, N, N, requires_grad=True),
              pos=torch.zeros(B, 2, N, N))
    no["dir_raw"][:, 1] = 1.0
    c = -49.0 + 2.0 * torch.arange(N).float()
    no["pos"][:, 0] = c.view(N, 1)
    no["pos"][:, 1] = c.view(1, N)
    no["e"].requires_grad_(True)
    # GT: a straight lane along y at column 20, rows 10..39 (x = -29 + ..), a long link from the end to a node far ahead
    ex = torch.zeros(B, N, N)
    ex[:, 10:40, 20] = 1
    ex[:, 45, 30] = 1
    uo, ui = torch.zeros(B, N, N), torch.zeros(B, N, N)
    uo[:, 39, 20], ui[:, 45, 30] = 1, 1
    lk = -torch.ones(B, 40, 4, dtype=torch.long)
    for i in range(10, 39):
        lk[:, i - 10] = torch.tensor([i, 20, i + 1, 20])
    lk[:, 29] = torch.tensor([39, 20, 45, 30])
    gt = dict(exists=ex, uout=uo, uin=ui, lk=lk)
    # make the first node of the long link point at the second: set the directions of the GT nodes along +x (rows) for the straight lane
    no["dir_raw"][:, 0], no["dir_raw"][:, 1] = 1.0, 0.0
    cen = torch.rand(B, 1, 100, 100)
    m = LinkHead(d=d, hidden=32)
    o = m(no, cen, gt=gt, K=128, KL=8)
    assert o["gt_missing"] == 0, o["gt_missing"]
    assert o["label"].sum() >= 2 * 30 - 1, o["label"].sum()                                   # every GT link (and nothing but GT pairs that are GT links) is labelled
    assert o["is_long"].any() and (o["sa"][~o["is_long"]] == 1 / 3).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(o["logit"], o["label"]) + o["sa"].mean()
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead
    assert no["tok"].grad is not None and float(no["tok"].grad.abs().sum()) > 0
    print("pairs", len(o["pb"]), "labelled", int(o["label"].sum()), "long", int(o["is_long"].sum()))
    print("qlinks selftest ok")


if __name__ == "__main__":
    _selftest()

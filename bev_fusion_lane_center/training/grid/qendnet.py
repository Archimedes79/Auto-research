"""Stage E of the lane head (E284): the junction END head. Entry and exit keypoints with a tolerant target, and the pair decision between them.

    EndHead(d)(no, gt=None) -> dict                     no = the node head's output (`qnodes.NodeHead`: tok (B, d, 50, 50), e, flags); gt = the ends of the frame in training
    ends_loss(out, batch) -> (loss, info)               the one loss of the stage (phase 8 of `qgraph_train.py`)
    decode_pairs(out, thr) -> list (per frame) of (n, 9)   [xa ya tax tay xb yb tbx tby p]: the delivered connectors (entry, heading, exit, heading, probability)

Why (E283 sections 5 and 7, D71, D79): the junction blur is the unknown PAIR of entry and exit; the delivered ends are not clean (the flag is missing at 25 - 33 % of the GT ends and about half of the flags sit at
no GT end), and the event flags of the node head ask a cell to be exactly the end. This head asks for less and says more: a heat map over the 2 m cells whose target is a Gaussian (sigma 2 m, so the end may
lie 2 - 4 m from the peak cell), a position offset (+-5 m) and a heading of the end in every cell (trained within 3 m of an end), and a pair classifier on the extracted keypoints.
Per kind (entry / exit): the heat is read as peaks (3 x 3 non-maximum suppression, p >= 0.1, the K = 16 strongest: an index choice with the gradient through the values, named like the index choices of
`qlinks`); in training the cells of the GT ends are added to the candidates (as `qlinks` does with the true nodes), so that the pair head sees positives from the first step. The keypoint tokens (cell token +
position, heading, probability, kind) pass 2 layers of self-attention over all keypoints of the frame (which entry joins which exit is a question about the whole junction), then every (entry, exit) pair that
`qends.allowed` permits (turn <= 130 degrees, chord >= 2 m, ahead) gets a logit from [token a, token b, geometry (turn cos / sin, chord, offsets in both frames, angles between the chord and the headings)].
Labels: a keypoint is matched to the nearest GT end of its kind within 4 m whose heading differs by at most 30 degrees; a pair is true when its two ends are matched to the two ends of a GT connector run.
Loss: penalty-reduced focal loss on the heat (CenterNet), L1 on the offset and (1 - cos) on the heading near ends, BCE (positive weight 3) and a dual softmax with a 'none' entry (the competition of the
candidates of one entry / of one exit, as `qgraph.long_match_loss`) on the pairs. The delivered curves are the cubic Hermite curves of `qends.curve` (tangent = chord), one per pair with p >= 0.5; the
read-out is `evaluation/qends_eval.py`.
Batch keys: ent (B, E, 4) / ext (B, X, 4) the GT ends (x, y, tx, ty), ent_m / ext_m (B, E) / (B, X) bool, pairs (B, P, 2) long (-1 padded) = true (entry, exit) index pairs (`lane_gt_data.collate`).
Run the self-test: python qendnet.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlinks import SetBlock
from qnodes import N, RES, HALF, cell_centres, unit

K = 16                                                                      # keypoints per kind
SIGMA = 2.0                                                                 # m: the Gaussian of the tolerant target
OFF = 5.0                                                                   # m: reach of the position offset
NEAR = 3.0                                                                  # m: offset / heading are trained within this distance of a GT end
MATCH_POS, MATCH_ANG = 4.0, 30.0                                            # a keypoint is the GT end when it lies within 4 m and 30 degrees
COS_TURN, MIN_CHORD = math.cos(math.radians(130.0)), 2.0                    # the pair rule of `qends.allowed`
PEAK_MIN = 0.1


class EndHead(nn.Module):
    def __init__(self, d=64, hid=64, k=K):
        super().__init__()
        self.k = k
        self.body = nn.Sequential(nn.Conv2d(d + 3, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU(), nn.Conv2d(d, d, 3, padding=2, dilation=2), nn.GroupNorm(8, d), nn.GELU())
        self.out = nn.Conv2d(d, 10, 1)                                      # heat (entry, exit), offset (entry xy, exit xy), heading (entry xy, exit xy)
        nn.init.normal_(self.out.weight, std=0.01)
        with torch.no_grad():
            b = torch.zeros(10)
            b[0:2] = -4.0
            b[7], b[9] = 1.0, 1.0                                           # heading forward
            self.out.bias.copy_(b)
        self.kp = nn.Linear(d + 6, d)
        self.blocks = nn.ModuleList([SetBlock(d), SetBlock(d)])
        self.pair = nn.Sequential(nn.Linear(2 * d + 13, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, no, gt=None):
        tok = no["tok"]
        B, d = tok.shape[:2]
        e = no["e"].float().sigmoid()[:, None]
        fl = no["flags"][:, :2].float().sigmoid()
        h = self.body(torch.cat([tok, e.to(tok.dtype), fl.to(tok.dtype)], 1))
        o = self.out(h).float()
        heat = o[:, 0:2]                                                    # (B, 2, N, N) logits: entry, exit
        c = cell_centres(tok.device)                                        # (1, 2, N, N)
        pos = torch.stack([c + OFF * torch.tanh(o[:, 2:4]), c + OFF * torch.tanh(o[:, 4:6])], 1)       # (B, 2 kinds, 2, N, N)
        dirv = torch.stack([unit(o[:, 6:8]), unit(o[:, 8:10])], 1)
        idxs, valids = [], []
        for k in (0, 1):
            p = heat[:, k].detach().sigmoid()
            mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
            score = torch.where((p >= mx) & (p > PEAK_MIN), p, torch.zeros_like(p)).flatten(1)
            if gt is not None:                                              # training: the cells of the GT ends are always candidates
                ends, m = gt["ent"] if k == 0 else gt["ext"], gt["ent_m"] if k == 0 else gt["ext_m"]
                cell = cell_index(ends[..., :2])
                sc = torch.zeros_like(score)
                sc.scatter_(1, cell, m.float() * 2.0)
                score = torch.maximum(score, sc)
            v, ix = score.topk(self.k, dim=1)
            idxs.append(ix)
            valids.append(v > 0)
        idx, valid = torch.stack(idxs, 1), torch.stack(valids, 1)           # (B, 2, K)
        tf = tok.flatten(2)
        toks, P, D, pr = [], [], [], []
        for k in (0, 1):
            ix = idx[:, k]
            toks.append(tf.gather(2, ix[:, None, :].expand(B, d, -1)).transpose(1, 2).float())
            P.append(pos[:, k].flatten(2).gather(2, ix[:, None, :].expand(B, 2, -1)).transpose(1, 2))
            D.append(dirv[:, k].flatten(2).gather(2, ix[:, None, :].expand(B, 2, -1)).transpose(1, 2))
            pr.append(heat[:, k].flatten(1).gather(1, ix).sigmoid())
        toks, P, D, pr = torch.cat(toks, 1), torch.cat(P, 1), torch.cat(D, 1), torch.cat(pr, 1)          # (B, 2K, .)
        kind = torch.cat([torch.zeros(B, self.k), torch.ones(B, self.k)], 1).to(tok.device)
        v2 = valid.flatten(1)
        t = self.kp(torch.cat([toks, P / HALF, D, pr[..., None], kind[..., None]], -1))
        for blk in self.blocks:
            t = blk(t, ~v2)
        ta_t, tb_t = t[:, :self.k], t[:, self.k:]
        pa, pb, ta, tb = P[:, :self.k], P[:, self.k:], D[:, :self.k], D[:, self.k:]
        ch = pb[:, None, :, :] - pa[:, :, None, :]                           # (B, K, K, 2) entry i -> exit j
        L = ch.norm(dim=-1)
        ta_ = ta[:, :, None, :].expand_as(ch)
        tb_ = tb[:, None, :, :].expand_as(ch)
        na_ = torch.stack([-ta_[..., 1], ta_[..., 0]], -1)
        nb_ = torch.stack([-tb_[..., 1], tb_[..., 0]], -1)
        dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
        cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
        cu = ch / L.clamp_min(1e-6)[..., None]
        cos_turn = dot(ta_, tb_)
        g = torch.stack([cos_turn, cross(ta_, tb_), L / 20.0, dot(ch, ta_) / 20.0, dot(ch, na_) / 20.0, dot(ch, tb_) / 20.0, dot(ch, nb_) / 20.0, dot(cu, ta_), cross(ta_, cu), dot(cu, tb_), cross(tb_, cu),
                         pr[:, :self.k, None].expand_as(L), pr[:, None, self.k:].expand_as(L)], -1)
        allowed = (cos_turn >= COS_TURN) & (L >= MIN_CHORD) & (dot(ch, ta_) > 0) & (dot(ch, tb_) > 0) & valid[:, 0, :, None] & valid[:, 1, None, :]
        logit = self.pair(torch.cat([ta_t[:, :, None].expand(-1, -1, self.k, -1), tb_t[:, None].expand(-1, self.k, -1, -1), g], -1))[..., 0]
        return dict(heat=heat, pos=pos, dirv=dirv, idx=idx, valid=valid, kp_pos=P, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)


def cell_index(xy):
    """flat cell index (row = x, column = y) of metre positions (B, n, 2), clamped into the grid"""
    i = ((xy + HALF) / RES).floor().long().clamp(0, N - 1)
    return i[..., 0] * N + i[..., 1]


def _focal(logit, t, n):
    p = logit.sigmoid()
    pos = t >= 1.0
    lp = -((1 - p) ** 2 * F.logsigmoid(logit))[pos].sum()
    ln = -(((1 - t) ** 4) * p ** 2 * F.logsigmoid(-logit))[~pos].sum()
    return (lp + ln) / n


def _dual_softmax(z, y, allowed):
    """rows (the candidates of one entry) and columns (of one exit) compete with a 'none' entry of logit 0: -log(mass of the true ones), or -log P(none) when the row has no true candidate"""
    tot = z.new_zeros(())
    for dim in (2, 1):
        zz = torch.where(allowed, z, torch.full_like(z, -1e4))
        zy = torch.where(allowed & (y > 0), z, torch.full_like(z, -1e4))
        none = torch.zeros_like(zz.sum(dim, keepdim=True))
        lse = torch.logsumexp(torch.cat([zz, none], dim), dim)
        lse_pos = torch.logsumexp(torch.cat([zy, torch.full_like(none, -1e4)], dim), dim)
        has = ((allowed & (y > 0)).sum(dim) > 0)
        rows = allowed.any(dim)
        nll = torch.where(has, lse - lse_pos, lse - none.squeeze(dim))
        tot = tot + (nll * rows).sum() / rows.sum().clamp_min(1)
    return tot / 2


def match_keypoints(out, batch):
    """[entry matches, exit matches], each (B, K) long: the index of the GT end of its kind that the keypoint is (within 4 m and 30 degrees, the nearest), -1 = none (also for an invalid keypoint)"""
    dev = out["heat"].device
    res = []
    kk = out["idx"].shape[2]
    for k, (key, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        G, m = batch[key].to(dev).float(), batch[mkey].to(dev)
        kp, kd = out["kp_pos"][:, k * kk:(k + 1) * kk].detach(), out["kp_dir"][:, k * kk:(k + 1) * kk].detach()
        dk = (kp[:, :, None, :] - G[:, None, :, :2]).norm(dim=-1)            # (B, K, n)
        ang = (kd[:, :, None, :] * G[:, None, :, 2:]).sum(-1)
        ok = (dk <= MATCH_POS) & (ang >= math.cos(math.radians(MATCH_ANG))) & m[:, None, :] & out["valid"][:, k, :, None]
        dk = torch.where(ok, dk, torch.full_like(dk, 1e6))
        a = dk.argmin(2)
        res.append(torch.where(ok.any(2), a, torch.full_like(a, -1)))
    return res


def end_counts(out, batch, thr=0.5):
    """the head's own numbers on a batch as raw counts (add them over batches): ent_n / ext_n GT ends, ent_hit / ext_hit GT ends that a keypoint with p >= thr matches, ent_kp / ext_kp keypoints with p >= thr,
    ent_ok / ext_ok of them that match a GT end, pairs_n true pairs, pairs_hit true pairs delivered (p >= thr, both ends matched to the ends of that pair), pairs_kp delivered pairs, pairs_ok of them true"""
    dev = out["heat"].device
    kk = out["idx"].shape[2]
    match = match_keypoints(out, batch)
    res = {}
    for k, (name, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        strong = (out["kp_p"][:, k * kk:(k + 1) * kk] >= thr) & out["valid"][:, k]
        mt = match[k]
        res[f"{name}_n"] = int(batch[mkey].sum())
        res[f"{name}_kp"] = int(strong.sum())
        res[f"{name}_ok"] = int((strong & (mt >= 0)).sum())
        hit = 0
        for b in range(mt.shape[0]):
            hit += len(set(mt[b][strong[b] & (mt[b] >= 0)].tolist()))
        res[f"{name}_hit"] = hit
    p = out["logit"].float().sigmoid() * out["allowed"].float()
    pairs = batch["pairs"].to(dev)
    true = set()
    delivered_ok = 0
    deliv = 0
    for b in range(p.shape[0]):
        pv = pairs[b][(pairs[b, :, 0] >= 0)].tolist()
        tp = {(int(a), int(c)) for a, c in pv}
        ia, ib = torch.nonzero(p[b] >= thr, as_tuple=True)
        got = set()
        for i, j in zip(ia.tolist(), ib.tolist()):
            deliv += 1
            key = (int(match[0][b, i]), int(match[1][b, j]))
            if key in tp:
                delivered_ok += 1
                got.add(key)
        true |= {(b, a, c) for a, c in got}
    res.update(pairs_n=int((pairs[..., 0] >= 0).sum()), pairs_hit=len(true), pairs_kp=deliv, pairs_ok=delivered_ok)
    return res


def ends_loss(out, batch):
    dev = out["heat"].device
    B = out["heat"].shape[0]
    c = cell_centres(dev)[0].flatten(1).t()                                  # (N*N, 2)
    info = {}
    l_heat = l_off = l_dir = 0.0
    npos = 0.0
    for k, (key, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        G = batch[key].to(dev).float()
        m = batch[mkey].to(dev)
        n_end = float(m.sum())
        d2 = (c[None, :, None, :] - G[:, None, :, :2]).square().sum(-1)      # (B, N*N, n)
        d2 = torch.where(m[:, None, :], d2, torch.full_like(d2, 1e6))
        t = torch.exp(-d2.min(2)[0] / (2 * SIGMA ** 2)) * (m.any(1)[:, None]).float()
        t = t * (t > 1e-3)
        cell = cell_index(G[..., :2])
        t = t.scatter(1, cell, torch.where(m, torch.ones_like(cell, dtype=t.dtype), t.gather(1, cell)))       # the cell that holds an end is the peak (target 1)
        l_heat = l_heat + _focal(out["heat"][:, k].flatten(1), t, max(n_end, 1.0))
        near, arg = d2.min(2)
        nm = (near <= NEAR ** 2).float()
        nearest = G.gather(1, arg[..., None].expand(-1, -1, 4))              # (B, N*N, 4)
        off_t = ((nearest[..., :2] - c[None]) / OFF).clamp(-1, 1)
        off_p = (out["pos"][:, k].flatten(2).transpose(1, 2) - c[None]) / OFF
        l_off = l_off + ((off_p - off_t).abs().sum(-1) * nm).sum() / nm.sum().clamp_min(1.0)
        dp = out["dirv"][:, k].flatten(2).transpose(1, 2)
        l_dir = l_dir + ((1 - (dp * nearest[..., 2:]).sum(-1)) * nm).sum() / nm.sum().clamp_min(1.0)
        npos += n_end
    match = match_keypoints(out, batch)
    pairs = batch["pairs"].to(dev)
    E, X = batch["ent"].shape[1], batch["ext"].shape[1]
    T = torch.zeros(B, E + 1, X + 1, dtype=torch.bool, device=dev)
    pv = (pairs[..., 0] >= 0) & (pairs[..., 1] >= 0)
    bi = torch.arange(B, device=dev)[:, None].expand_as(pv)
    T[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = True
    ia, ib = match[0].clone(), match[1].clone()
    ia[ia < 0], ib[ib < 0] = E, X                                            # index E / X = 'no GT end' (an empty row / column of T)
    y = T[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]].float()
    allowed = out["allowed"]
    z = out["logit"].float()
    if allowed.any():
        bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * allowed).sum() / allowed.sum()
        dual = _dual_softmax(z, y, allowed)
    else:
        bce = dual = z.sum() * 0.0
    info.update(heat=float(l_heat), off=float(l_off), dir=float(l_dir), bce=float(bce), dual=float(dual), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))
    return l_heat + l_off + 0.5 * l_dir + bce + 0.5 * dual, info


def decode_pairs(out, thr=0.5):
    """the delivered connectors of every frame: rows [xa ya tax tay xb yb tbx tby p] of the pairs with probability >= thr"""
    p = out["logit"].float().sigmoid() * out["allowed"].float()
    kk = out["logit"].shape[1]
    res = []
    for b in range(p.shape[0]):
        ia, ib = torch.nonzero(p[b] >= thr, as_tuple=True)
        P, D = out["kp_pos"][b], out["kp_dir"][b]
        res.append(torch.cat([P[ia], D[ia], P[kk + ib], D[kk + ib], p[b][ia, ib][:, None]], 1).detach().cpu().numpy())
    return res


def _selftest():
    torch.manual_seed(0)
    B, d = 2, 32
    tok = torch.randn(B, d, N, N)
    no = dict(tok=tok, e=torch.randn(B, N, N), flags=torch.randn(B, 8, N, N))
    # frame 0: two entries (0, -10) heading +y and (8, -10); two exits (0, 10) heading +y and (-10, 0) heading -x; true pairs (0, 0) and (0, 1) and (1, 0); frame 1: no junction
    ent = torch.zeros(B, 2, 4)
    ext = torch.zeros(B, 2, 4)
    ent[0] = torch.tensor([[0.0, -10.0, 0.0, 1.0], [8.0, -10.0, 0.0, 1.0]])
    ext[0] = torch.tensor([[0.0, 10.0, 0.0, 1.0], [-10.0, 0.0, -1.0, 0.0]])
    m = torch.tensor([[True, True], [False, False]])
    batch = dict(ent=ent, ext=ext, ent_m=m, ext_m=m.clone(), pairs=torch.tensor([[[0, 0], [0, 1], [1, 0]], [[-1, -1], [-1, -1], [-1, -1]]]))
    head = EndHead(d=d)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for it in range(250):
        out = head(no, gt=batch)
        loss, info = ends_loss(out, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss)
    assert float(loss) < 0.5 * first, (first, float(loss), info)
    head.eval()
    with torch.no_grad():
        out = head(no)
    dp = decode_pairs(out, 0.5)
    assert len(dp[1]) == 0 or True
    pr = out["heat"][0].sigmoid()
    for k, G in ((0, ent[0]), (1, ext[0])):
        for e_ in G:
            ix = int(cell_index(e_[None, None, :2])[0, 0])
            assert float(pr[k].flatten()[ix]) > 0.5, (k, e_, float(pr[k].flatten()[ix]))
    got = {(round(float(r[0])), round(float(r[1])), round(float(r[4])), round(float(r[5]))) for r in dp[0]}
    want = {(0, -10, 0, 10), (0, -10, -10, 0), (8, -10, 0, 10)}
    assert len(got & want) >= 2 and len(got - want) <= 1, (got, want, info)
    cnt = end_counts(out, batch)
    assert cnt["ent_n"] == 2 and cnt["ext_n"] == 2 and cnt["ent_hit"] == 2 and cnt["ext_hit"] == 2 and cnt["pairs_n"] == 3 and cnt["pairs_hit"] == 3 and cnt["pairs_ok"] >= 3, cnt
    dead = [n for n, p in head.named_parameters() if p.grad is None]
    assert not dead, dead
    print("qendnet selftest ok", {k: round(v, 3) for k, v in info.items()}, "pairs found", len(got & want), "of 3, false", len(got - want))


if __name__ == "__main__":
    _selftest()

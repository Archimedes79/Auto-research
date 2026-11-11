"""The junction pair head v2 (E287, D82): which entry joins which exit, read from the junction maps of E286 (`jmaps.py`), with the order rule and the painted / lane evidence along the candidate curve.

    JEnds(d=64, order=True, evidence=True)(jout, P, gt=None) -> dict    jout = `JModel` output (j1, j05, f1), P = the pyramid dict (centre1 for the lane heat); gt = the batch's GT ends in training
    jends_loss(out, batch) -> (loss, info)                              pair BCE + dual softmax with a 'none' entry (as `qendnet`) + the soft crossing penalty (order rule)
    The `out` dict carries the keys of `qendnet.EndHead` (heat, idx, valid, kp_pos, kp_dir, kp_p, logit, allowed), so `qendnet.end_counts` / `match_keypoints` and the read-out of `evaluation/qends_eval.py` work unchanged.

What is new against the E284 end head (which reads node tokens): (1) keypoints from the 1 m entry / exit heat of the junction maps (3 x 3 peaks, soft-argmax position over the 3 x 3 window: differentiable in the heat; K = 32 per kind, the GT end cells are added in training); (2) `order`: the rank of an
entry in its entry group and of an exit in its exit group (groups as `jgt._groups`: heading within 25 degrees, 6 m along / across) and the group sizes enter the pair logit, and a soft crossing penalty
punishes P(i -> j) P(i' -> j') when the ranks of i, i' and j, j' are crossed inside one pair of groups (D82: entry 1 to exit 1, entry 2 to exit 2); (3) `evidence`: the Hermite curve of the pair (tangent = chord) is
sampled at 12 points in the maps (lane-centre probability of B, zone, conflict core, stop line, crossing) and the means / minimum are features. All soft; the choice of the K strongest peaks is an index choice (the
gradient flows through the values, the D7 convention), the group / rank bookkeeping is a function of the keypoint positions (no gradient needed), both named in the card.
Run the self-test: python jends.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jgt as JG
import qendnet as QEN
from qlinks import SetBlock

K = 32
HALF, RES, N1 = 50.0, 1.0, 100
PEAK_MIN = 0.1
NG, NEV, NRK = 13, 6, 5
COS_TURN, MIN_CHORD = QEN.COS_TURN, QEN.MIN_CHORD


def _centres(device):
    c = -HALF + (torch.arange(N1, device=device).float() + 0.5) * RES
    return c


def _unit(v, eps=1e-6):
    return v / v.norm(dim=1, keepdim=True).clamp_min(eps)


def _hermite(a, ta, b, tb, n=12):
    """points (..., n, 2) of the cubic Hermite curve a -> b (tangent length = chord)"""
    L = (b - a).norm(dim=-1, keepdim=True)
    s = torch.linspace(0, 1, n, device=a.device).view(*([1] * (a.dim() - 1)), n, 1)
    a_, b_, ta_, tb_ = a[..., None, :], b[..., None, :], (L * ta)[..., None, :], (L * tb)[..., None, :]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a_ + h10 * ta_ + h01 * b_ + h11 * tb_


def _sample(maps, pts):
    """maps (B, C, H, W) (rows = x, columns = y, origin -50 m, extent 100 m), pts (B, ..., 2) metres -> (B, ..., C) bilinear samples (zero outside)"""
    B = maps.shape[0]
    shp = pts.shape[1:-1]
    g = torch.stack([pts[..., 1], pts[..., 0]], -1).reshape(B, -1, 1, 2) / HALF            # grid x = column = y, grid y = row = x
    out = F.grid_sample(maps.float(), g.float(), mode="bilinear", padding_mode="zeros", align_corners=False)      # (B, C, n, 1)
    return out[..., 0].transpose(1, 2).reshape(B, *shp, maps.shape[1])


class JEnds(nn.Module):
    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64):
        super().__init__()
        self.k, self.order, self.evidence = k, order, evidence
        self.kp = nn.Linear(f1 + 6, d)
        self.blocks = nn.ModuleList([SetBlock(d), SetBlock(d)])
        nin = 2 * d + NG + (NEV if evidence else 0) + (NRK if order else 0)
        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, jout, P, gt=None):
        j1, f1 = jout["j1"].float(), jout["f1"]
        B, dev = j1.shape[0], j1.device
        heat = j1[:, 2:4]                                                        # entry, exit logits (B, 2, 100, 100)
        dirv = torch.stack([_unit(j1[:, 4:6]), _unit(j1[:, 6:8])], 1)            # (B, 2, 2, H, W)
        c = _centres(dev)
        idxs, valids = [], []
        for kk in (0, 1):
            p = heat[:, kk].detach().sigmoid()
            mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
            score = torch.where((p >= mx) & (p > PEAK_MIN), p, torch.zeros_like(p)).flatten(1)
            if gt is not None:
                ends, m = gt["ent"] if kk == 0 else gt["ext"], gt["ent_m"] if kk == 0 else gt["ext_m"]
                i = ((ends[..., :2] + HALF) / RES).floor().long().clamp(0, N1 - 1)
                cell = i[..., 0] * N1 + i[..., 1]
                sc = torch.zeros_like(score)
                sc.scatter_reduce_(1, cell, m.float() * 2.0, "amax", include_self=True)
                score = torch.maximum(score, sc)
            v, ix = score.topk(self.k, dim=1)
            idxs.append(ix)
            valids.append(v > 0)
        idx, valid = torch.stack(idxs, 1), torch.stack(valids, 1)                # (B, 2, K)
        # soft-argmax position over the 3 x 3 window of the peak (differentiable in the heat), heading and probability at the peak, features of the 1 m map
        pos, D, pr, feat = [], [], [], []
        off = torch.tensor([-1.0, 0.0, 1.0], device=dev)
        dx, dy = torch.meshgrid(off, off, indexing="ij")
        dx, dy = dx.flatten(), dy.flatten()
        for kk in (0, 1):
            ix = idx[:, kk]                                                      # (B, K)
            pm = heat[:, kk].sigmoid()
            unf = F.unfold(pm[:, None], 3, padding=1)                           # (B, 9, H*W), neighbour order row-major (dx slowest)
            w = unf.gather(2, ix[:, None, :].expand(B, 9, -1))                  # (B, 9, K)
            wn = w / w.sum(1, keepdim=True).clamp_min(1e-6)
            i0, j0 = ix // N1, ix % N1
            x = c[i0] + (wn * dx[None, :, None] * RES).sum(1)
            y = c[j0] + (wn * dy[None, :, None] * RES).sum(1)
            pos.append(torch.stack([x, y], -1))
            D.append(dirv[:, kk].flatten(2).gather(2, ix[:, None, :].expand(B, 2, -1)).transpose(1, 2))
            pr.append(pm.flatten(1).gather(1, ix))
            feat.append(f1.flatten(2).gather(2, ix[:, None, :].expand(B, f1.shape[1], -1)).transpose(1, 2).float())
        pos, D, pr, feat = torch.cat(pos, 1), torch.cat(D, 1), torch.cat(pr, 1), torch.cat(feat, 1)       # (B, 2K, .)
        K_ = self.k
        kind = torch.cat([torch.zeros(B, K_), torch.ones(B, K_)], 1).to(dev)
        v2 = valid.flatten(1)
        t = self.kp(torch.cat([feat, pos / HALF, D, pr[..., None], kind[..., None]], -1))
        for blk in self.blocks:
            t = blk(t, ~v2)
        ta_t, tb_t = t[:, :K_], t[:, K_:]
        pa, pb, ta, tb = pos[:, :K_], pos[:, K_:], D[:, :K_], D[:, K_:]
        ch = pb[:, None, :, :] - pa[:, :, None, :]
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
                         pr[:, :K_, None].expand_as(L), pr[:, None, K_:].expand_as(L)], -1)
        allowed = (cos_turn >= COS_TURN) & (L >= MIN_CHORD) & (dot(ch, ta_) > 0) & (dot(ch, tb_) > 0) & valid[:, 0, :, None] & valid[:, 1, None, :]
        feats = [ta_t[:, :, None].expand(-1, -1, K_, -1), tb_t[:, None].expand(-1, K_, -1, -1), g]
        if self.evidence:
            pts = _hermite(pa[:, :, None, :].expand_as(ch), ta_, pb[:, None, :, :].expand_as(ch), tb_)             # (B, K, K, 12, 2)
            lane = _sample(P["centre1"][:, :1].float().sigmoid(), pts)[..., 0]
            zc = _sample(jout["j1"][:, :2].float().sigmoid(), pts)
            pnt = _sample(jout["j05"].float().sigmoid(), pts)
            ev = torch.stack([lane.mean(-1), lane.min(-1)[0], zc[..., 0].mean(-1), zc[..., 1].mean(-1), pnt[..., 0].max(-1)[0], pnt[..., 1].max(-1)[0]], -1)
            feats.append(ev)
        ga = ra = gb = rb = None
        if self.order:
            rk, ga, ra, gb, rb = self.rank_matrix(pos, D, valid)
            feats.append(rk)
        logit = self.pair(torch.cat(feats, -1))[..., 0]
        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)
        if self.order:
            out.update(ga=ga, ra=ra, gb=gb, rb=rb)
        return out

    @torch.no_grad()
    def rank_matrix(self, pos, D, valid):
        """(B, K, K, 5) rank features and the group ids / ranks (B, K) of entries (ga, ra) and exits (gb, rb); the groups of `jgt._groups` over the valid keypoints of each kind"""
        B, K_ = valid.shape[0], self.k
        dev = pos.device
        ga = np.zeros((B, K_), int)
        gb = np.zeros((B, K_), int)
        ra = np.zeros((B, K_), int)
        rb = np.zeros((B, K_), int)
        ka = np.ones((B, K_))
        kb = np.ones((B, K_))
        pn, dn, vn = pos.cpu().numpy(), D.cpu().numpy(), valid.cpu().numpy()
        for b in range(B):
            for kind, (g_, r_, k_) in enumerate(((ga, ra, ka), (gb, rb, kb))):
                v = vn[b, kind]
                ids = np.where(v)[0]
                if not len(ids):
                    g_[b, :] = -1 - np.arange(K_)
                    continue
                E = np.concatenate([pn[b, kind * K_ + ids], dn[b, kind * K_ + ids]], 1)
                grp = JG._groups(E)
                rk = JG._ranks(E, grp)
                g_[b, :] = -1 - np.arange(K_)                                   # an invalid keypoint is its own group
                g_[b, ids], r_[b, ids] = grp, rk
                cnt = np.bincount(grp)
                k_[b, ids] = cnt[grp]
        ga_t, gb_t = torch.from_numpy(ga).to(dev), torch.from_numpy(gb).to(dev)
        ra_t, rb_t = torch.from_numpy(ra).float().to(dev), torch.from_numpy(rb).float().to(dev)
        ka_t, kb_t = torch.from_numpy(ka).float().to(dev), torch.from_numpy(kb).float().to(dev)
        rin = (ra_t / (ka_t - 1).clamp_min(1.0))[:, :, None].expand(B, K_, K_)
        rout = (rb_t / (kb_t - 1).clamp_min(1.0))[:, None, :].expand(B, K_, K_)
        same = ((ka_t[:, :, None] == kb_t[:, None, :]) & (ra_t[:, :, None] == rb_t[:, None, :])).float()
        rk = torch.stack([rin, rout, ka_t[:, :, None].expand(B, K_, K_) / 4.0, kb_t[:, None, :].expand(B, K_, K_) / 4.0, same], -1)
        return rk, ga_t, ra_t, gb_t, rb_t


def crossing_penalty(out):
    """the soft order penalty: sum over (i, j), (i', j') of P(i -> j) P(i' -> j') where i, i' share an entry group, j, j' share an exit group, and the ranks cross (rank i < rank i' but rank j > rank j');
    P = sigmoid(logit) on the allowed pairs; normalised by the mass of P"""
    z = out["logit"].float()
    Pm = z.sigmoid() * out["allowed"].float()
    ga, ra, gb, rb = out["ga"], out["ra"], out["gb"], out["rb"]
    Rin = ((ga[:, :, None] == ga[:, None, :]) & (ra[:, :, None] < ra[:, None, :])).float()          # [i, i'] same entry group, rank i < rank i'
    Rout = ((gb[:, :, None] == gb[:, None, :]) & (rb[:, :, None] > rb[:, None, :])).float()         # [j, j'] same exit group, rank j > rank j'
    t = torch.bmm(torch.bmm(Rin, Pm), Rout.transpose(1, 2))                                          # (B, K, K): sum over i', j' of Rin[i, i'] P[i', j'] Rout[j, j']
    return (Pm * t).sum() / (Pm.sum() + 1.0)


def jends_loss(out, batch, cross_w=1.0):
    dev = out["heat"].device
    B = out["heat"].shape[0]
    match = QEN.match_keypoints(out, batch)
    pairs = batch["pairs"].to(dev)
    E, X = batch["ent"].shape[1], batch["ext"].shape[1]
    T = torch.zeros(B, E + 1, X + 1, dtype=torch.bool, device=dev)
    pv = (pairs[..., 0] >= 0) & (pairs[..., 1] >= 0)
    bi = torch.arange(B, device=dev)[:, None].expand_as(pv)
    T[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = True
    ia, ib = match[0].clone(), match[1].clone()
    ia[ia < 0], ib[ib < 0] = E, X
    y = T[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]].float()
    allowed = out["allowed"]
    z = out["logit"].float()
    info = {}
    if allowed.any():
        bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * allowed).sum() / allowed.sum()
        dual = QEN._dual_softmax(z, y, allowed)
    else:
        bce = dual = z.sum() * 0.0
    loss = bce + 0.5 * dual
    info.update(bce=float(bce), dual=float(dual), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))
    if "ga" in out and cross_w > 0 and allowed.any():
        cp = crossing_penalty(out)
        loss = loss + cross_w * cp
        info["cross"] = float(cp)
    return loss, info


def _selftest():
    torch.manual_seed(0)
    B, d = 2, 64
    jout = dict(j1=torch.randn(B, 8, 100, 100) * 0.1 - 2.0, j05=torch.randn(B, 2, 200, 200) - 2.0, f1=torch.randn(B, d, 100, 100))
    jout["j1"][:, 4], jout["j1"][:, 6] = 0.0, 0.0                              # heading (0, 1) = north for entries and exits
    jout["j1"][:, 5], jout["j1"][:, 7] = 5.0, 5.0
    P = dict(centre1=torch.randn(B, 12, 100, 100))
    # two GT entries side by side going north into a junction, two exits going north: pairs 0 -> 0, 1 -> 1
    ent = torch.tensor([[[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0]]] * B)
    gt = dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))
    batch = dict(ent=ent, ext=ext, ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * B))
    for order, evidence in ((False, False), (True, False), (True, True)):
        m = JEnds(d=d, order=order, evidence=evidence)
        out = m(jout, P, gt=gt)
        assert out["logit"].shape == (B, K, K) and out["kp_pos"].shape == (B, 2 * K, 2), out["logit"].shape
        loss, info = jends_loss(out, batch)
        loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None]
        assert not dead, (order, evidence, dead)
        assert info["pos_pairs"] >= 2 * B, info
    # the crossing penalty: P = 1 on the crossed pair set {0 -> 1, 1 -> 0} is large, on the ordered set {0 -> 0, 1 -> 1} it is zero
    out = dict(logit=torch.full((1, K, K), -20.0), allowed=torch.ones(1, K, K, dtype=torch.bool), ga=torch.zeros(1, K, dtype=torch.long), gb=torch.zeros(1, K, dtype=torch.long),
               ra=torch.arange(K).float()[None], rb=torch.arange(K).float()[None])
    out["logit"][0, 0, 0], out["logit"][0, 1, 1] = 20.0, 20.0
    assert float(crossing_penalty(out)) < 1e-3
    out["logit"][0, 0, 0], out["logit"][0, 1, 1] = -20.0, -20.0
    out["logit"][0, 0, 1], out["logit"][0, 1, 0] = 20.0, 20.0
    assert float(crossing_penalty(out)) > 0.3, float(crossing_penalty(out))
    print("jends selftest ok; params", sum(p.numel() for p in JEnds().parameters()) / 1e6, "M")


if __name__ == "__main__":
    _selftest()

"""The junction pair head v2 (E287, D82): which entry joins which exit, read from the junction maps of E286 (`jmaps.py`), with the order rule and the painted / lane evidence along the candidate curve.

    JEnds(d=64, order=True, evidence=True)(jout, P, gt=None) -> dict    jout = `JModel` output (j1, j05, f1), P = the pyramid dict (centre1 for the lane heat); gt = the batch's GT ends in training
    jends_loss(out, batch) -> (loss, info)                              pair BCE + dual softmax with a 'none' entry (as `qendnet`) + the soft crossing penalty (order rule)
    The `out` dict carries the keys of `qendnet.EndHead` (heat, idx, valid, kp_pos, kp_dir, kp_p, logit, allowed), so `qendnet.end_counts` / `match_keypoints` and the read-out of `evaluation/qends_eval.py` work unchanged.

What is new against the E284 end head (which reads node tokens): (1) keypoints from the 1 m entry / exit heat of the junction maps (3 x 3 peaks, soft-argmax position over the 3 x 3 window: differentiable in the heat; K = 32 per kind, the GT end cells are added in training); (2) `order`: the rank of an
entry in its entry group and of an exit in its exit group (groups as `jgt._groups`: heading within 25 degrees, 6 m along / across) and the group sizes enter the pair logit, and a soft crossing penalty
punishes P(i -> j) P(i' -> j') when the ranks of i, i' and j, j' are crossed inside one pair of groups (D82: entry 1 to exit 1, entry 2 to exit 2); (3) `evidence`: the Hermite curve of the pair (tangent = chord) is
sampled at 12 points in the maps (lane-centre probability of B, zone, conflict core, stop line, crossing) and the means / minimum are features. All soft; the choice of the K strongest peaks is an index choice (the
gradient flows through the values, the D7 convention), the group / rank bookkeeping is a function of the keypoint positions (no gradient needed), both named in the card.
E302 `capacity`: instead of independent pair logits (+ dual softmax) the pair probability comes from the plan of a log-domain Sinkhorn with a dustbin: every entry has a predicted out-degree {0, 1, 2, 3+} (0 = a false keypoint), every exit an in-degree; an entry that
feeds a turn AND the straight exit (27 % of the entries, E285) has mass 2, a merge exit takes two entries (33 %). A plan entry is a MASS (up to the degree), the pair probability is q = P / max(1, min(row mass, column mass)) in [0, 1] (the share of the smaller capacity this pair uses; one entry of degree 1 with two equally good exits gives 0.5 each), read as a logit, so the read-out (`logit >= 0`, i.e. q >= 0.5), the counts and the crossing penalty work unchanged. Differentiable (10 unrolled iterations).
E308 `ntok`: B's node tokens (the frozen node head of the E273 baseline on the same pyramid, 64 channels at 2 m) are sampled at the keypoints and enter the keypoint token: the E284 end head reads them and ranks the pairs between true ends better (AUC 0.974) than the head on the maps' own features alone (0.930, E287 p2).
E317 `geom_loss` (`--geom-w`): the delivered connector is supervised directly: the curve of every true keypoint pair (`_bezier_s`, the keypoint ends, headings and learned handle scales) against the GT connector polyline of the pair (symmetric point-to-polyline
distance, smooth L1 1 m) plus the position of every matched keypoint against its GT end. The gradient reaches the soft-argmax positions (so the heat of the maps head), the headings and the scales: the shape is a constraint of the loss, not only of the parametrisation.
E318 `unc`: a position scale (log sigma, metres) per keypoint from its token; the end-position term becomes the Laplace NLL |d| exp(-s) + s (heteroscedastic: an end may be declared uncertain); s enters the pair logit as a detached feature and is the output `kp_logsig` (an uncertainty per end for the planner).
Both are off by default (the E312 recipe is unchanged bit for bit). Run the self-test: python jends.py
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
SINK_ITERS = 10
DEG_VALUES = (0.0, 1.0, 2.0, 3.3)                                                                          # the degree classes 0, 1, 2, 3+ as masses


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


def _bezier_s(a, ta, b, tb, sa=None, sb=None, n=16):
    """points (..., n, 2) of the DELIVERED connector, the cubic Bezier of `qends.curve_s` in torch: control points a + sa L ta and b - sb L tb with L = the chord and the handle scales sa, sb (the E303 scale head; None = 1/3, which is `_hermite`'s chord-tangent curve);
    differentiable in the end points, the headings and the scales"""
    L = (b - a).norm(dim=-1, keepdim=True)
    sa_ = 1.0 / 3.0 if sa is None else sa[..., None]
    sb_ = 1.0 / 3.0 if sb is None else sb[..., None]
    s = torch.linspace(0, 1, n, device=a.device).view(*([1] * (a.dim() - 1)), n, 1)
    a_, b_ = a[..., None, :], b[..., None, :]
    q1, q2 = (a + sa_ * L * ta)[..., None, :], (b - sb_ * L * tb)[..., None, :]
    return (1 - s) ** 3 * a_ + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b_


def _dist_to_polyline(pts, poly):
    """(..., Np) distance of every point of `pts` (..., Np, 2) to the polyline `poly` (..., Nq, 2) (point-to-segment, so the sampling of the polyline does not enter); a small floor under the square root keeps the gradient finite on the line"""
    a, b = poly[..., :-1, :], poly[..., 1:, :]
    ab = b - a
    ap = pts[..., :, None, :] - a[..., None, :, :]
    t = ((ap * ab[..., None, :, :]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)[..., None, :]).clamp(0.0, 1.0)
    d2 = ((ap - t[..., None] * ab[..., None, :, :]) ** 2).sum(-1)
    return (d2.min(-1)[0] + 1e-8).sqrt()


def _huber(d, delta=1.0):
    """smooth L1 of a distance: quadratic inside `delta` metres, linear beyond"""
    return torch.where(d < delta, 0.5 * d * d / delta, d - 0.5 * delta)


def _sample(maps, pts):
    """maps (B, C, H, W) (rows = x, columns = y, origin -50 m, extent 100 m), pts (B, ..., 2) metres -> (B, ..., C) bilinear samples (zero outside)"""
    B = maps.shape[0]
    shp = pts.shape[1:-1]
    g = torch.stack([pts[..., 1], pts[..., 0]], -1).reshape(B, -1, 1, 2) / HALF            # grid x = column = y, grid y = row = x
    out = F.grid_sample(maps.float(), g.float(), mode="bilinear", padding_mode="zeros", align_corners=False)      # (B, C, n, 1)
    return out[..., 0].transpose(1, 2).reshape(B, *shp, maps.shape[1])


def sinkhorn_plan(z, allowed, re, rx, alpha, iters=SINK_ITERS, full=False):
    """(B, K, K) transport plan of the pair scores `z` (B, K, K; disallowed pairs get ~no mass) with a dustbin row and column: row masses `re` (B, K) = the expected out-degree of the entries, column masses `rx` = the in-degree of the exits;
    the dustbin takes what is not connected (its mass makes both marginal sums equal). Log domain, `iters` unrolled iterations, differentiable in z, re, rx and alpha. P_ij is a mass (<= min(re_i, rx_j)); `full`: the whole (B, K + 1, K + 1) plan with the dustbin row / column."""
    B, K_ = z.shape[0], z.shape[1]
    S = torch.where(allowed, z, torch.full_like(z, -30.0))
    col = alpha.expand(B, K_, 1)
    M = torch.cat([torch.cat([S, col], 2), torch.cat([alpha.expand(B, 1, K_), z.new_zeros(B, 1, 1)], 2)], 1)             # (B, K + 1, K + 1)
    log_a = torch.cat([(re + 1e-6).log(), (rx.sum(1, keepdim=True) + 1e-6).log()], 1)
    log_b = torch.cat([(rx + 1e-6).log(), (re.sum(1, keepdim=True) + 1e-6).log()], 1)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(M + v[:, None, :], 2)
        v = log_b - torch.logsumexp(M + u[:, :, None], 1)
    plan = (M + u[:, :, None] + v[:, None, :]).exp()
    return plan if full else plan[:, :K_, :K_]


ATT_M, ATT_REACH, ATT_ANG, ATT_TGT = 8, 4.0, 45.0, 2.5


class Attach(nn.Module):
    """E320: a differentiable replacement of the snap of the read-out (E311). Every keypoint (entry or exit) chooses among the nodes of B's frozen node head (`qnodes.NodeHead` outputs on the same batch, 2 m grid: existence, the 8 type flags, long-link
    and section scores, position, direction, token) that lie within ATT_REACH m and ATT_ANG degrees of it (the reach of the delivered wiring, `qends_eval.WIRE_POS / WIRE_ANG`), at most ATT_M of them (the nearest). Score = an MLP on (distance, along / across offset and heading agreement of the node, its existence and type
    probabilities, its token, the kind of the keypoint, the keypoint's own token); a learned 'no attachment' (the dustbin, from the keypoint token) competes in the softmax. The attached end is the weighted mean: position = w0 x keypoint + sum w_m x node_m, heading = the
    normalised weighted mean of the headings, a convex combination of the keypoint and the candidate nodes: when one weight dominates the curve starts / ends at that node, and it falls back to the keypoint when nothing is in reach. It is NOT guaranteed to be a delivered piece end (the candidates are the raw nodes with existence > 0.05, mid-section nodes of a lane included; the weighted mean lies inside their hull; the dustbin falls back to the keypoint), and the proximity wiring of the read-out (4 m / 45 degrees, `qends_eval.add_connectors`) still decides which pieces the connector is joined to. All soft; the choice of the candidate set (the nearest ATT_M gated nodes) is an index choice, the
    gradient flows through the values and the keypoint position (the D7 convention, named in the card). Returns att_pos / att_dir (B, 2K, 2), att_p = 1 - w0 (B, 2K), att_logit (B, 2K, M + 1; index 0 = no attachment), att_cpos (B, 2K, M, 2) the candidates' positions, att_cvalid (B, 2K, M) bool."""

    def __init__(self, d=64, tok=64, hid=64, m=ATT_M):
        super().__init__()
        self.m = m
        self.tp, self.kpj = nn.Linear(tok, 32), nn.Linear(d, 32)
        self.mlp = nn.Sequential(nn.Linear(5 + 12 + 1 + 32 + 32, hid), nn.GELU(), nn.Linear(hid, 1))
        self.dust = nn.Linear(d, 1)

    def forward(self, t, pos, D, kind, nodes):
        B, n = t.shape[0], t.shape[1]
        npos = nodes["pos"].float().flatten(2).transpose(1, 2)                                            # (B, 2500, 2) metres
        ndir = nodes["dir_raw"].float().flatten(2).transpose(1, 2)
        ndir = ndir / ndir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pe = nodes["e"].float().sigmoid().flatten(1)                                                      # (B, 2500)
        nf = torch.cat([pe[..., None], nodes["flags"].float().sigmoid().flatten(2).transpose(1, 2), torch.stack([nodes["uout"], nodes["uin"], nodes["s"]], 1).float().sigmoid().flatten(2).transpose(1, 2)], -1)      # (B, 2500, 12)
        ntok = nodes["tok"].float().flatten(2).transpose(1, 2)                                            # (B, 2500, tok)
        with torch.no_grad():                                                                             # the candidate set: the nearest ATT_M nodes within reach whose direction agrees and that exist at all
            dist = (npos[:, None, :, :] - pos.detach()[:, :, None, :]).norm(dim=-1)                       # (B, 2K, 2500)
            ca = (ndir[:, None, :, :] * D.detach()[:, :, None, :]).sum(-1)
            gate = (dist <= ATT_REACH) & (ca >= math.cos(math.radians(ATT_ANG))) & (pe[:, None, :] > 0.05)
            key = torch.where(gate, dist, torch.full_like(dist, float("inf")))
            val, idx = key.topk(self.m, dim=2, largest=False)                                             # (B, 2K, M)
            cval = torch.isfinite(val)
        g2 = lambda x: x.gather(1, idx.reshape(B, -1, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])).reshape(B, n, self.m, *x.shape[2:])  # noqa: E731
        cpos, cdir, cf, ct = g2(npos), g2(ndir), g2(nf), g2(ntok)                                          # candidates (B, 2K, M, .)
        rel = cpos - pos[:, :, None, :]                                                                   # grad through the keypoint position
        nrm = torch.stack([-D[..., 1], D[..., 0]], -1)[:, :, None, :]
        dd = (rel ** 2).sum(-1).add(1e-8).sqrt()
        feat = torch.cat([torch.stack([dd / ATT_REACH, (rel * D[:, :, None, :]).sum(-1) / ATT_REACH, (rel * nrm).sum(-1) / ATT_REACH, (cdir * D[:, :, None, :]).sum(-1), (cdir[..., 0] * D[:, :, None, 1] - cdir[..., 1] * D[:, :, None, 0])], -1), cf,
                          kind[:, :, None, None].expand(B, n, self.m, 1).float(), self.tp(ct), self.kpj(t)[:, :, None, :].expand(B, n, self.m, 32)], -1)
        lc = self.mlp(feat)[..., 0].float().masked_fill(~cval, float("-inf"))                              # (B, 2K, M)
        logits = torch.cat([self.dust(t)[..., 0:1].float(), lc], -1)                                       # index 0 = no attachment
        w = logits.softmax(-1)
        w0, wc = w[..., 0:1], w[..., 1:]
        apos = w0 * pos + (wc[..., None] * cpos).sum(2)
        adir = w0 * D + (wc[..., None] * cdir).sum(2)
        adir = adir / adir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return dict(att_pos=apos, att_dir=adir, att_p=1.0 - w0[..., 0], att_logit=logits, att_cpos=cpos, att_cvalid=cval)


class JEnds(nn.Module):
    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, capacity=False, ntok=0, unc=False, attach=False):
        super().__init__()
        self.k, self.order, self.evidence, self.scales, self.capacity, self.ntok, self.unc = k, order, evidence, scales, capacity, ntok, unc
        self.kp = nn.Linear(f1 + 6 + ntok, d)
        self.blocks = nn.ModuleList([SetBlock(d), SetBlock(d)])
        nin = 2 * d + NG + (NEV if evidence else 0) + (NRK if order else 0) + (2 if unc else 0)                  # E318: + the position scales (log sigma) of the entry and the exit keypoint
        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        if capacity:                                                                                         # E302: the degree heads on the keypoint tokens (classes 0, 1, 2, 3+) and the dustbin score
            self.deg_e, self.deg_x = nn.Linear(d, 4), nn.Linear(d, 4)
            self.alpha = nn.Parameter(torch.zeros(()))
        if scales:                                                                                           # E303: the two handle scales of the delivered curve, in chords (0.05 .. 1.25 like the fit grid)
            self.scale = nn.Sequential(nn.Linear(nin, 32), nn.GELU(), nn.Linear(32, 2))
        if unc:                                                                                              # E318: one position scale per keypoint (log sigma in metres) from its self-attended token; sigma = 1 m at the start
            self.logsig = nn.Linear(d, 1)
            nn.init.zeros_(self.logsig.weight)
            nn.init.zeros_(self.logsig.bias)
        self.attach = Attach(d=d) if attach else None                                                        # E320: the attached connector ends (B's nodes as candidates), needs the node head's outputs in `forward(nodes=)`

    def forward(self, jout, P, gt=None, tok=None, nodes=None):
        j1, f1 = jout["j1"].float(), jout["f1"]
        B, dev = j1.shape[0], j1.device
        heat = j1[:, 2:4]                                                        # entry, exit logits (B, 2, 100, 100)
        dirv = torch.stack([_unit(j1[:, 4:6]), _unit(j1[:, 6:8])], 1)            # (B, 2, 2, H, W)
        c = _centres(dev)
        idxs, valids, reals = [], [], []
        for kk in (0, 1):
            p = heat[:, kk].detach().sigmoid()
            mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
            score = torch.where((p >= mx) & (p > PEAK_MIN), p, torch.zeros_like(p)).flatten(1)
            peak_score = score
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
            reals.append(peak_score.gather(1, ix) > 0)                                    # a keypoint that is a real peak of the heat (not only an injected GT cell)
        idx, valid, real = torch.stack(idxs, 1), torch.stack(valids, 1), torch.stack(reals, 1)                # (B, 2, K)
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
        if self.ntok:                                                                                          # E308: B's node tokens at the keypoints (bilinear, 2 m grid over the same 100 m)
            feat = torch.cat([feat, _sample(tok.float(), pos)], -1)
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
            rk, ga, ra, gb, rb = self.rank_matrix(pos, D, valid & real if gt is not None else valid)         # groups and ranks from the real peaks only: the same distribution in training and at evaluation
            feats.append(rk)
        ls = None
        if self.unc:                                                                                         # E318: the position scales enter the pair logit as features, detached (the pair loss does not shape them)
            ls = self.logsig(t)[..., 0].float().clamp(-3.0, 3.0)                                             # (B, 2K): sigma 0.05 .. 20 m
            lsd = ls.detach()
            feats.append(torch.stack([lsd[:, :K_, None].expand_as(L), lsd[:, None, K_:].expand_as(L)], -1))
        fcat = torch.cat(feats, -1)
        logit = self.pair(fcat)[..., 0]
        extra = {}
        if self.capacity:                                                                                    # E302: the pair probability is the plan of a capacitated Sinkhorn, expressed as a logit
            de, dgx = self.deg_e(ta_t), self.deg_x(tb_t)
            vals = logit.new_tensor(DEG_VALUES)
            re = (de.float().softmax(-1) * vals).sum(-1) * valid[:, 0]
            rx = (dgx.float().softmax(-1) * vals).sum(-1) * valid[:, 1]
            Pm = sinkhorn_plan(logit.float(), allowed, re, rx, self.alpha)
            Pm = (Pm / torch.minimum(re[:, :, None], rx[:, None, :]).clamp_min(1.0)).clamp(1e-5, 1 - 1e-5)          # the share of the smaller capacity
            logit = (Pm / (1 - Pm)).log()
            extra = dict(deg_e=de, deg_x=dgx)
        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed, **extra)
        if self.scales:
            out["scale"] = 0.05 + 1.2 * self.scale(fcat).sigmoid()
        if ls is not None:
            out["kp_logsig"] = ls
        if self.attach is not None:
            assert nodes is not None, "JEnds(attach=True) needs the outputs of B's node head (`nodes=`)"
            out.update(self.attach(t, pos, D, kind, nodes))
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


def degree_classes(T, match_e, match_x):
    """(cls_e, cls_x) (B, K) long: the degree class {0, 1, 2, 3+} of every keypoint: the number of true pairs of the GT end it is matched to (rows of T (B, E, X) = out-degree of an entry, columns = in-degree of an exit), 0 for an unmatched keypoint"""
    res = []
    for deg, mt in ((T.sum(2), match_e), (T.sum(1), match_x)):
        res.append(torch.where(mt >= 0, deg.gather(1, mt.clamp_min(0)).clamp(max=3).long(), torch.zeros_like(mt)))
    return res


def geom_loss(out, batch, y, allowed, ia, ib, bi, pv, pairs):
    """E317 / E318: the geometry of the delivered connector, supervised directly. (1) SHAPE: for every true pair whose two ends are keypoints (y = 1, allowed) the delivered curve (`_bezier_s`: the keypoint ends, the headings, the learned
    handle scales, 16 samples) against the GT connector polyline of the pair (`batch['ppts']`, 16 points of equal arc length): symmetric point-to-polyline distance, smooth L1 with delta 1 m, curve -> GT and GT -> curve. (2) ENDS: every valid keypoint
    within 4 m / 30 degrees of a GT end of its kind against the nearest such end, smooth L1 (E317) or, when the head has a position scale (`kp_logsig`, E318), the Laplace NLL |d| exp(-s) + s (the net may say that an end is uncertain, at the price of s).
    The gradient reaches the keypoint positions (through the soft-argmax into the heat, so into the maps head with --e2e), the headings (the dir maps) and the scales. Returns (loss or None, info); `ia`, `ib`: the matched GT end of every keypoint
    (E / X = none) as in `jends_loss` (they pick the GT polyline of a keypoint pair).
    E320 (`att_pos` in out): the curve and the end term use the ATTACHED ends (`Attach`), and the attachment is trained by a cross entropy over the candidates + 'no attachment' (target: the candidate node nearest to the GT end of the keypoint, within ATT_TGT m)."""
    dev = out["heat"].device
    B, K_ = out["valid"].shape[0], out["idx"].shape[2]
    E, X = batch["ent"].shape[1], batch["ext"].shape[1]
    pp = batch["ppts"].to(dev).float()
    PP = torch.full((B, E + 1, X + 1) + tuple(pp.shape[2:]), float("nan"), device=dev)
    PP[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = pp[pv]
    PPt = PP[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]                      # (B, K, K, N, 2): the GT polyline of the pair (keypoint i, keypoint j), NaN when it is no true pair
    m = (y > 0) & allowed & torch.isfinite(PPt[..., 0, 0])
    kp, kd = out["kp_pos"].float(), out["kp_dir"].float()
    ep, ed = (out["att_pos"].float(), out["att_dir"].float()) if "att_pos" in out else (kp, kd)               # E320: the FINAL ends of the delivered curve are the attached ones
    info, loss = {}, None
    if m.any():
        b_, i_, j_ = m.nonzero(as_tuple=True)
        sc = out.get("scale")
        curve = _bezier_s(ep[b_, i_], ed[b_, i_], ep[b_, K_ + j_], ed[b_, K_ + j_], None if sc is None else sc[b_, i_, j_, 0].float(), None if sc is None else sc[b_, i_, j_, 1].float())
        gt = PPt[m]
        d1, d2 = _dist_to_polyline(curve, gt), _dist_to_polyline(gt, curve)
        shape = _huber(d1).mean() + _huber(d2).mean()
        loss = shape
        info.update(geom_shape=float(shape), geom_n=float(m.sum()), geom_d_curve=float(d1.mean()), geom_d_gt=float(d2.mean()))
    ends, n_end, d_all = kp.new_zeros(()), 0, []
    ce_tot, st = kp.new_zeros(()), dict(n=0, ce=0.0, acc=0, near=0, bin=0, kp_d=[], att_d=[])                    # st: the attachment statistics over the keypoints with at least one candidate (critic 2026-11-12: the trivial dustbin cases and the keypoints without a node in reach are no information)
    for k, (key, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        G, gm = batch[key].to(dev).float(), batch[mkey].to(dev)
        kpk, kdk = kp[:, k * K_:(k + 1) * K_], kd[:, k * K_:(k + 1) * K_]
        with torch.no_grad():                                                                                # every valid keypoint of the kind is pulled to the nearest GT end within 4 m / 30 degrees (many to one: a real peak that
            dk = (kpk[:, :, None, :] - G[:, None, :, :2]).norm(dim=-1)                                        # lies next to the keypoint the BCE matched is a delivered connector end too, so its position counts)
            ang = (kdk[:, :, None, :] * G[:, None, :, 2:]).sum(-1)
            okm = (dk <= QEN.MATCH_POS) & (ang >= math.cos(math.radians(QEN.MATCH_ANG))) & gm[:, None, :] & out["valid"][:, k, :, None]
            a_ = torch.where(okm, dk, torch.full_like(dk, 1e6)).argmin(2)
            ok = okm.any(2)
        if not ok.any():
            continue
        g = G[..., :2].gather(1, a_[..., None].expand(-1, -1, 2))
        epk = ep[:, k * K_:(k + 1) * K_]
        d = ((epk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok]
        if "att_logit" in out:                                                                               # E320: the attachment target of a keypoint = the node of B's head nearest to the GT end it belongs to (within ATT_TGT m), else 'no attachment'
            lg = out["att_logit"][:, k * K_:(k + 1) * K_]
            with torch.no_grad():
                dc = ((out["att_cpos"][:, k * K_:(k + 1) * K_] - g[:, :, None, :]) ** 2).sum(-1).sqrt()
                dc = torch.where(out["att_cvalid"][:, k * K_:(k + 1) * K_], dc, torch.full_like(dc, 1e6))
                tgt = torch.where(dc.min(2)[0] <= ATT_TGT, dc.argmin(2) + 1, torch.zeros_like(dc[..., 0], dtype=torch.long))
            ce_tot = ce_tot + F.cross_entropy(lg[ok], tgt[ok], reduction="sum")
            with torch.no_grad():
                sel = ok & out["att_cvalid"][:, k * K_:(k + 1) * K_].any(-1)
                if sel.any():
                    st["n"] += int(sel.sum())
                    st["ce"] += float(F.cross_entropy(lg[sel], tgt[sel], reduction="sum"))
                    st["acc"] += int((lg[sel].argmax(-1) == tgt[sel]).sum())
                    st["near"] += int((tgt[sel] == 1).sum())                                                  # the baseline 'take the nearest candidate' (the candidates are sorted by distance)
                    st["bin"] += int((tgt[sel] == 0).sum())
                    st["kp_d"].append(((kpk - g) ** 2).sum(-1).add(1e-8).sqrt()[sel])
                    st["att_d"].append(((epk - g) ** 2).sum(-1).add(1e-8).sqrt()[sel])
        if "kp_logsig" in out:
            s = out["kp_logsig"][:, k * K_:(k + 1) * K_][ok]
            ends = ends + (d * torch.exp(-s) + s).sum()
            info["geom_sig"] = float(s.exp().mean())
        else:
            ends = ends + _huber(d).sum()
        n_end += int(ok.sum())
        d_all.append(d.detach())
    if n_end:
        ends = ends / n_end
        loss = ends if loss is None else loss + ends
        info.update(geom_end=float(ends), geom_end_d=float(torch.cat(d_all).mean()))
        if "att_logit" in out:                                                                               # E320: the cross entropy of the attachment over the candidates (+ the dustbin), normalised like the end term
            loss = loss + ce_tot / n_end
            if st["n"]:                                                                                      # logged over the keypoints with >= 1 candidate: att_acc against att_acc_nearest (= take the nearest candidate), att_ce (chance is ln(n_candidates + 1)), the attached against the keypoint end error
                info.update(att_n=float(st["n"]), att_ce=st["ce"] / st["n"], att_acc=st["acc"] / st["n"], att_acc_nearest=st["near"] / st["n"], att_bin=st["bin"] / st["n"],
                            att_kp_d=float(torch.cat(st["kp_d"]).mean()), att_d=float(torch.cat(st["att_d"]).mean()))
    return loss, info


def jends_loss(out, batch, cross_w=1.0, driven_w=0.0, scale_w=0.5, geom_w=0.0):
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
    wmat = torch.ones_like(y)
    if driven_w > 0 and "pdrv" in batch:                                                                   # E298: the positive terms of the pairs the ego drives count 1 + driven_w times
        Dv = torch.zeros(B, E + 1, X + 1, device=dev)
        pd = batch["pdrv"].to(dev)
        Dv[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = pd[pv]
        yd = Dv[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]
        wmat = 1.0 + driven_w * yd * y
    if allowed.any():
        bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * wmat * allowed).sum() / allowed.sum()
        dual = QEN._dual_softmax(z, y, allowed)
    else:
        bce = dual = z.sum() * 0.0
    cap = "deg_e" in out
    loss = bce + (0.0 if cap else 0.5 * dual)                                                               # the plan already makes rows and columns compete (E302)
    info.update(bce=float(bce), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))
    if not cap:
        info["dual"] = float(dual)
    if cap:                                                                                                  # E302: the degree classes of the keypoints (a keypoint that is no GT end: class 0)
        ce_tot, acc_tot, n_tot = z.new_zeros(()), 0.0, 0
        cls_all = degree_classes(T[:, :E, :X].float(), match[0], match[1])
        for k_, (cls, dl) in enumerate(zip(cls_all, (out["deg_e"], out["deg_x"]))):
            v = out["valid"][:, k_]
            if v.any():
                ce_tot = ce_tot + F.cross_entropy(dl.float()[v], cls[v])
                acc_tot += float((dl[v].argmax(-1) == cls[v]).float().mean())
                n_tot += 1
        if n_tot:
            loss = loss + 0.5 * ce_tot
            info.update(deg_ce=float(ce_tot), deg_acc=acc_tot / n_tot)
    if "scale" in out and "pscale" in batch and scale_w > 0:                                                    # E303: L1 of the predicted handle scales against the fitted ones, on the matched true pairs
        S = torch.full((B, E + 1, X + 1, 2), float("nan"), device=dev)
        ps = batch["pscale"].to(dev)
        S[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = ps[pv]
        St = S[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]
        m = (y > 0) & allowed & torch.isfinite(St[..., 0])
        if m.any():
            ls = (out["scale"][m] - St[m]).abs().mean()
            loss = loss + scale_w * ls
            info["scale_l1"] = float(ls)
    if "ga" in out and cross_w > 0 and allowed.any():
        cp = crossing_penalty(out)
        loss = loss + cross_w * cp
        info["cross"] = float(cp)
    if geom_w > 0:                                                                                              # E317 / E318 / E320: the shape and the ends of the delivered connector
        assert "ppts" in batch, "geom_w > 0 needs the GT connector polylines in the batch (`QData.DEFAULT_PPTS = True`, set by jends_train.py --geom-w): without them the whole geometry and attachment loss would be skipped silently"
        lg, ig = geom_loss(out, batch, y, allowed, ia, ib, bi, pv, pairs)
        if lg is not None:
            loss = loss + geom_w * lg
            info.update(ig)
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
        if order and evidence:                                                                         # E298: the driven weight raises the positive BCE terms and changes nothing when off
            batch["pdrv"] = torch.ones(B, 2)
            _, i0 = jends_loss(out, batch, driven_w=0.0)
            _, i2 = jends_loss(out, batch, driven_w=2.0)
            assert i2["bce"] > i0["bce"] + 1e-6 and abs(i0["bce"] - info["bce"]) < 1e-6, (i0, i2, info)
            batch["pdrv"] = torch.zeros(B, 2)
            _, iz = jends_loss(out, batch, driven_w=2.0)
            assert abs(iz["bce"] - i0["bce"]) < 1e-6, (iz, i0)
    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)
    mt = JEnds(d=d, order=True, evidence=False, ntok=64)
    tok_a, tok_b = torch.randn(B, 64, 50, 50), torch.randn(B, 64, 50, 50)
    la = mt(jout, P, gt=gt, tok=tok_a)["logit"]
    lb = mt(jout, P, gt=gt, tok=tok_b)["logit"]
    assert float((la - lb).abs().max()) > 1e-4, "the node tokens must reach the pair logits"
    loss, info = jends_loss(mt(jout, P, gt=gt, tok=tok_a), batch)
    loss.backward()
    assert all(p_.grad is not None for p_ in mt.parameters()), [n for n, p_ in mt.named_parameters() if p_.grad is None]
    # E302: the capacitated plan. Two entries, two exits, all degree 1, scores favour the diagonal: the plan is the identity; an entry of degree 2 feeds two exits of degree 1; a false keypoint (degree 0) gets no mass
    z = torch.full((1, 4, 4), -3.0)
    z[0, 0, 0], z[0, 1, 1] = 4.0, 4.0
    al = torch.zeros(())
    Pl = sinkhorn_plan(z, torch.ones(1, 4, 4, dtype=torch.bool), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), al)
    assert Pl[0, 0, 0] > 0.8 and Pl[0, 1, 1] > 0.8 and Pl[0, 0, 1] < 0.1 and Pl[0, 2:].sum() < 1e-3 and Pl[0, :, 2:].sum() < 1e-3, Pl[0]
    z2 = torch.full((1, 4, 4), -3.0)
    z2[0, 0, 0], z2[0, 0, 1] = 2.0, 2.0
    P2 = sinkhorn_plan(z2, torch.ones(1, 4, 4, dtype=torch.bool), torch.tensor([[2.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), al)
    assert P2[0, 0, 0] > 0.7 and P2[0, 0, 1] > 0.7, P2[0]
    # critic 2026-11-12: pairs 0 -> 0, 0 -> 1, 1 -> 2, 2 -> 2 (entry 0 has degree 2, exit 2 a merge of two): the class targets, with an unmatched keypoint
    Tm = torch.zeros(1, 3, 3)
    for a_, b_ in ((0, 0), (0, 1), (1, 2), (2, 2)):
        Tm[0, a_, b_] = 1.0
    ce_, cx_ = degree_classes(Tm, torch.tensor([[0, 1, 2, -1]]), torch.tensor([[0, 1, 2, -1]]))
    assert ce_.tolist() == [[2, 1, 1, 0]] and cx_.tolist() == [[1, 1, 2, 0]], (ce_, cx_)
    # the full plan has the marginals it was given, also when the total entry mass is not the total exit mass (E != X), and the share of the smaller capacity is in [0, 1]
    ze = torch.randn(1, 5, 5)
    re_, rx_ = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0]])
    full = sinkhorn_plan(ze, torch.ones(1, 5, 5, dtype=torch.bool), re_, rx_, torch.zeros(()), iters=50, full=True)
    assert abs(float(full[0, 0].sum()) - 2.0) < 1e-2 and abs(float(full[0, 1].sum()) - 1.0) < 1e-2 and abs(float(full[0, 5].sum()) - 2.5) < 1e-2, full[0].sum(1)
    assert abs(float(full[0, :, 0].sum()) - 1.0) < 1e-2 and abs(float(full[0, :, 5].sum()) - 3.0) < 1e-2, full[0].sum(0)
    mc = JEnds(d=d, order=True, evidence=True, capacity=True)
    out = mc(jout, P, gt=gt)
    assert out["deg_e"].shape == (B, K, 4) and torch.isfinite(out["logit"]).all()
    loss, info = jends_loss(out, batch)
    loss.backward()
    dead = [n for n, p_ in mc.named_parameters() if p_.grad is None]
    assert not dead and "deg_ce" in info and info["pos_pairs"] >= 2 * B, (dead, info)
    # critic 2026-11-11: with no real peak at all (heat far below the peak floor) every injected GT keypoint is its own group: the rank features cannot see the injected ends
    jq = dict(jout, j1=jout["j1"].clone())
    jq["j1"][:, 2:4] = -9.0
    mq = JEnds(d=d, order=True, evidence=False)
    oq = mq(jq, P, gt=gt)
    ga = oq["ga"]
    assert all(len(set(ga[b].tolist())) == K for b in range(B)), "injected keypoints must not form groups"
    # E303: the scale head (two handle scales per pair) trains against the fitted scales of the true pairs
    m = JEnds(d=d, order=True, evidence=True, scales=True)
    out = m(jout, P, gt=gt)
    assert out["scale"].shape == (B, K, K, 2) and float(out["scale"].min()) >= 0.05 and float(out["scale"].max()) <= 1.25
    batch["pscale"] = torch.full((B, 2, 2), 0.3)
    loss, info = jends_loss(out, batch)
    loss.backward()
    assert "scale_l1" in info and all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.scale.parameters()), info
    batch.pop("pscale")
    _, info2 = jends_loss(out, batch)
    assert "scale_l1" not in info2
    # E317: the pieces of the geometry loss. The delivered curve with chord tangents (scales 1/3) is `_hermite`; a straight connector lies on its chord; distance / smooth L1 by hand
    pa_, pb_ = torch.tensor([[0.0, 0.0], [1.0, 2.0]]), torch.tensor([[0.0, 12.0], [9.0, 2.0]])
    ta_, tb_ = torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cb, ch = _bezier_s(pa_, ta_, pb_, tb_, n=12), _hermite(pa_, ta_, pb_, tb_, n=12)
    assert float((cb - ch).abs().max()) < 1e-5, "the Bezier of the delivered curve at scales 1/3 must be the chord-tangent Hermite curve"
    line = torch.stack([pa_, pb_], 1)                                                                          # (2, 2, 2): the straight polylines
    assert float(_dist_to_polyline(cb, line).max()) < 1e-3
    assert abs(float(_dist_to_polyline(torch.tensor([[0.0, 1.0], [3.0, 0.0]]), torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))[1]) - 2.0) < 1e-3
    assert abs(float(_huber(torch.tensor(0.5))) - 0.125) < 1e-6 and abs(float(_huber(torch.tensor(2.0))) - 1.5) < 1e-6
    # the loss in the model: the GT connector of the two pairs is the straight line between the GT ends (batch['ppts']); gradient into the heat (positions), the headings, the scales, the pair features
    try:                                                                                                       # critic 2026-11-12: geom_w > 0 without the GT polylines must fail loudly, not skip the whole geometry loss
        jends_loss(out, batch, geom_w=0.5)
        raise RuntimeError("geom_w > 0 without batch['ppts'] must raise")
    except AssertionError as ex_:
        assert "polylines" in str(ex_), ex_
    ppts = torch.zeros(B, 2, 16, 2)
    for k_, x_ in enumerate((-1.75, 1.75)):
        ppts[:, k_, :, 0], ppts[:, k_, :, 1] = x_, torch.linspace(-8.0, 8.0, 16)
    batch["ppts"] = ppts
    m = JEnds(d=d, order=True, evidence=True, scales=True)
    jr = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
    out = m(jr, P, gt=gt)
    l0, i0 = jends_loss(out, batch)
    lg, ig = jends_loss(out, batch, geom_w=0.5)
    assert "geom_shape" not in i0 and "geom_shape" in ig and ig["geom_n"] >= 2 * B and float(lg) > float(l0) and ig["geom_end_d"] >= 0, (i0, ig)
    lg.backward()
    g1 = jr["j1"].grad
    assert float(g1[:, 2:4].abs().sum()) > 0, "the geometry loss must reach the entry / exit heat (the soft-argmax positions)"
    assert float(g1[:, 4:8].abs().sum()) > 0, "the geometry loss must reach the heading maps"
    assert all(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in m.scale.parameters()), "the geometry loss must reach the scale head"
    # E318: a position scale per keypoint, features of the pair logit (detached), the Laplace NLL for the ends
    mu = JEnds(d=d, order=True, evidence=True, scales=True, unc=True)
    out = mu(jout, P, gt=gt)
    assert out["kp_logsig"].shape == (B, 2 * K) and float(out["kp_logsig"].abs().max()) < 1e-6, "sigma starts at 1 m"
    out["logit"].sum().backward()
    assert mu.logsig.weight.grad is None or float(mu.logsig.weight.grad.abs().sum()) == 0.0, "the pair loss must not shape the position scale (detached feature)"
    mu.zero_grad()
    out = mu(jout, P, gt=gt)
    lu, iu = jends_loss(out, batch, geom_w=0.5)
    lu.backward()
    assert "geom_sig" in iu and float(mu.logsig.bias.grad.abs().sum()) > 0, "the end NLL must train the position scale"
    # E320: the attached ends. B's node grid (2 m cells, 50 x 50): a node ON the lane before the junction near the entry (-1.0, -7.0), a decoy node 3.4 m from the GT entry, a node after the junction near the exit (-1.0, 9.0), nothing else exists
    cc = -50.0 + (torch.arange(50).float() + 0.5) * 2.0
    nodes = dict(e=torch.full((B, 50, 50), -8.0), flags=torch.full((B, 8, 50, 50), -8.0), uout=torch.full((B, 50, 50), -8.0), uin=torch.full((B, 50, 50), -8.0), s=torch.full((B, 50, 50), -8.0),
                 pos=torch.stack([cc.view(50, 1).expand(50, 50), cc.view(1, 50).expand(50, 50)], 0)[None].expand(B, -1, -1, -1).clone(), dir_raw=torch.zeros(B, 2, 50, 50), tok=torch.randn(B, 64, 50, 50))
    nodes["dir_raw"][:, 1] = 1.0
    for (i_, j_) in ((24, 21), (22, 21), (24, 29)):
        nodes["e"][:, i_, j_] = 4.0
    ma = JEnds(d=d, order=True, evidence=True, scales=True, attach=True)
    jr = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
    out = ma(jr, P, gt=gt, nodes=nodes)
    assert out["att_pos"].shape == (B, 2 * K, 2) and out["att_logit"].shape == (B, 2 * K, ATT_M + 1) and out["att_cvalid"].shape == (B, 2 * K, ATT_M), out["att_logit"].shape
    assert torch.allclose(out["att_logit"].softmax(-1).sum(-1), torch.ones(B, 2 * K), atol=1e-5)
    far = ~out["att_cvalid"].any(-1)                                                                          # a keypoint with no node in reach keeps its own position and heading
    assert far.any() and torch.allclose(out["att_pos"][far], out["kp_pos"][far], atol=1e-6) and float(out["att_p"][far].max()) < 1e-6, "no candidate in reach: the end stays the keypoint"
    assert (out["att_cvalid"].sum(-1) <= ATT_M).all() and float((out["att_dir"].norm(dim=-1) - 1).abs().max()) < 1e-5
    batch["ppts"] = ppts
    opt = torch.optim.Adam(ma.parameters(), lr=3e-3)
    first = None
    for it in range(150):
        out = ma(jout, P, gt=gt, nodes=nodes)
        lt, it_ = jends_loss(out, batch, geom_w=1.0)
        opt.zero_grad()
        lt.backward()
        opt.step()
        first = it_ if first is None else first
    assert it_["att_ce"] < 0.5 * first["att_ce"] and it_["att_acc"] >= 0.9, (first, it_)
    assert it_["geom_end_d"] < first["geom_end_d"] and it_["geom_end_d"] < 1.5, (first["geom_end_d"], it_["geom_end_d"])        # the attached ends are nearer to the GT ends than the keypoints were (att_kp_d)
    jr = dict(jout, j1=jout["j1"].clone().requires_grad_(True))
    ma.zero_grad()
    out = ma(jr, P, gt=gt, nodes=nodes)
    lt, _ = jends_loss(out, batch, geom_w=1.0)
    lt.backward()
    assert all(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in list(ma.attach.mlp.parameters()) + list(ma.attach.dust.parameters())), "the loss must reach the attachment scores and the dustbin"
    assert float(jr["j1"].grad[:, 2:4].abs().sum()) > 0, "the loss must reach the heat through the attached ends"
    # the NLL by hand: ends 3 m off with s = 0 -> d / sigma + s is large and wants a larger sigma; ends 0.1 m off wants a smaller one
    oq = dict(heat=torch.zeros(1, 2, 4, 4), valid=torch.ones(1, 2, 2, dtype=torch.bool), idx=torch.zeros(1, 2, 2, dtype=torch.long), kp_pos=torch.tensor([[[3.0, 0.0], [0.1, 0.0], [20.0, 0.0], [0.0, 0.0]]]),
              kp_dir=torch.tensor([[[1.0, 0.0]] * 3 + [[0.0, 1.0]]]), kp_logsig=torch.zeros(1, 4, requires_grad=True))
    bq = dict(ent=torch.tensor([[[0.0, 0.0, 1.0, 0.0]] * 2]), ext=torch.tensor([[[0.0, 0.0, 1.0, 0.0]] * 2]), ent_m=torch.ones(1, 2, dtype=torch.bool), ext_m=torch.ones(1, 2, dtype=torch.bool),
              ppts=torch.full((1, 0, 16, 2), float("nan")), pairs=-torch.ones(1, 0, 2, dtype=torch.long))
    lq, iq = geom_loss(oq, bq, torch.zeros(1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.bool), torch.tensor([[2, 2]]), torch.tensor([[2, 2]]), torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.bool), -torch.ones(1, 0, 2, dtype=torch.long))
    lq.backward()
    gs = oq["kp_logsig"].grad[0]
    assert gs[0] < 0 < gs[1] and gs[2] == 0 and gs[3] == 0, f"a 3 m error must push log sigma up, a 0.1 m error down, a keypoint 20 m away or turned 90 degrees is no end (no gradient): {gs}"
    assert abs(iq["geom_end_d"] - 1.55) < 1e-3, iq                                                              # mean of 3.0 and 0.1: the two keypoints that are ends
    # the crossing penalty: P = 1 on the crossed pair set {0 -> 1, 1 -> 0} is large, on the ordered set {0 -> 0, 1 -> 1} it is zero
    out = dict(logit=torch.full((1, K, K), -20.0), allowed=torch.ones(1, K, K, dtype=torch.bool), ga=torch.zeros(1, K, dtype=torch.long), gb=torch.zeros(1, K, dtype=torch.long),
               ra=torch.arange(K).float()[None], rb=torch.arange(K).float()[None])
    out["logit"][0, 0, 0], out["logit"][0, 1, 1] = 20.0, 20.0
    assert float(crossing_penalty(out)) < 1e-3
    out["logit"][0, 0, 0], out["logit"][0, 1, 1] = -20.0, -20.0
    out["logit"][0, 0, 1], out["logit"][0, 1, 0] = 20.0, 20.0
    assert float(crossing_penalty(out)) > 0.3, float(crossing_penalty(out))
    # E320, critic 2026-11-12 (three mutations survived the first self-test): the candidate gate, the number of candidates, the heading mix and the direct position gradient, on a scene built for them
    cg = -50.0 + (torch.arange(50).float() + 0.5) * 2.0
    n2 = dict(e=torch.full((1, 50, 50), -8.0), flags=torch.full((1, 8, 50, 50), -8.0), uout=torch.full((1, 50, 50), -8.0), uin=torch.full((1, 50, 50), -8.0), s=torch.full((1, 50, 50), -8.0),
              pos=torch.stack([cg.view(50, 1).expand(50, 50), cg.view(1, 50).expand(50, 50)], 0)[None].clone(), dir_raw=torch.zeros(1, 2, 50, 50), tok=torch.randn(1, 64, 50, 50))
    n2["dir_raw"][:, 1] = 1.0
    for i_ in (23, 24, 25):                                                                                   # a block of 12 existing nodes around the keypoint, all heading (0, 1)
        for j_ in (19, 20, 21, 22):
            n2["e"][:, i_, j_] = 4.0
    rot = lambda deg: (-math.sin(math.radians(deg)), math.cos(math.radians(deg)))  # noqa: E731
    n2["dir_raw"][0, :, 23, 21] = torch.tensor(rot(60.0))                                                   # 60 degrees off: outside the 45 degree gate
    n2["dir_raw"][0, :, 25, 21] = torch.tensor(rot(20.0))                                                   # 20 degrees off: inside the gate, it must turn the attached heading
    kpA, kpB = torch.tensor([-1.8, -7.9]), torch.tensor([30.0, 30.0])                                          # A has more than ATT_M gated nodes, B has none
    pos2 = torch.stack([kpA, kpB])[None].clone().requires_grad_(True)
    D2 = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    at = Attach(d=64)
    for p_ in at.parameters():
        nn.init.zeros_(p_)
    with torch.no_grad():
        at.dust.bias.fill_(-20.0)                                                                             # no dustbin: the weights are uniform over the valid candidates (zero scores)
    o2 = at(torch.zeros(1, 2, 64), pos2, D2, torch.zeros(1, 2), n2)
    npos2, ndir2 = n2["pos"][0].flatten(1).t(), torch.nn.functional.normalize(n2["dir_raw"][0].flatten(1).t(), dim=-1)
    dd2 = (npos2 - kpA).norm(dim=-1)
    gate2 = (dd2 <= ATT_REACH) & ((ndir2 * D2[0, 0]).sum(-1) >= math.cos(math.radians(ATT_ANG))) & (n2["e"][0].flatten() > -2.9)
    assert int(gate2.sum()) > ATT_M, int(gate2.sum())                                                         # more gated nodes than the cap: the nearest ATT_M must be kept
    want = npos2[gate2][dd2[gate2].argsort()[:ATT_M]]
    got = o2["att_cpos"][0, 0][o2["att_cvalid"][0, 0]]
    assert o2["att_cvalid"][0, 0].sum() == ATT_M and torch.allclose(got, want, atol=1e-5), (got, want)
    assert not any(torch.allclose(row, npos2[23 * 50 + 21]) for row in got), "the node turned by 60 degrees must be gated out"
    assert not o2["att_cvalid"][0, 1].any() and torch.allclose(o2["att_pos"][0, 1], kpB), "a keypoint with no node in reach keeps its position"
    mean_dir = torch.nn.functional.normalize(ndir2[gate2][dd2[gate2].argsort()[:ATT_M]].mean(0), dim=0)
    dev_deg = math.degrees(math.acos(float(torch.clamp((o2["att_dir"][0, 0] * D2[0, 0]).sum(), -1, 1))))
    assert float((o2["att_dir"][0, 0] - mean_dir).abs().max()) < 1e-4 and dev_deg > 1.0, (o2["att_dir"][0, 0], mean_dir, dev_deg)    # the 20 degree node turns the attached heading: ignoring the heading mix gives dev_deg = 0
    for p_ in at.parameters():                                                                               # the direct gradient of the attached position w.r.t. the keypoint: w0 per coordinate (the scores are constants here)
        nn.init.zeros_(p_)
    o3 = at(torch.zeros(1, 2, 64), pos2, D2, torch.zeros(1, 2), n2)
    o3["att_pos"].sum().backward()
    w0A = 1.0 / (ATT_M + 1.0)
    assert abs(float(pos2.grad[0, 0].sum()) - 2 * w0A) < 1e-5 and abs(float(pos2.grad[0, 1].sum()) - 2.0) < 1e-5, pos2.grad                  # w0 x keypoint carries a gradient (cut it with detach and this is 0)
    # flag-off identity that stays in the repo (critic 2026-11-12): the E312 pair head (the `je.*` weights of runs/e312_p_s0, when the run is there) on a seeded synthetic input gives the numbers pinned from the committed code of 7386e0e1 (E317 / E318 / E320 off); no attach / unc keys
    import os
    ckp = os.path.join(os.environ.get("BFLC_ROOT", ""), "runs", "e312_p_s0", "model.pt")
    if os.environ.get("BFLC_ROOT") and os.path.exists(ckp):
        st_ = {k_[3:]: v_ for k_, v_ in torch.load(ckp, map_location="cpu")["state"].items() if k_.startswith("je.")}
        mg = JEnds(d=64, order=True, evidence=True, scales=True)
        mg.load_state_dict(st_)
        mg.eval()
        gg = torch.Generator().manual_seed(7)
        jg_ = dict(j1=torch.randn(2, 8, 100, 100, generator=gg) * 0.5 - 2.5, j05=torch.randn(2, 2, 200, 200, generator=gg) - 2.0, f1=torch.randn(2, 64, 100, 100, generator=gg))
        pg_ = dict(centre1=torch.randn(2, 12, 100, 100, generator=gg))
        with torch.no_grad():
            og = mg(jg_, pg_, gt=None)
        for k_, v_ in dict(logit=-25719.9688, kp_pos=-645.8732, scale=2228.1379, kp_p=34.751).items():
            assert abs(float(og[k_].sum()) - v_) <= 1e-3 * max(1.0, abs(v_)), (k_, float(og[k_].sum()), v_)
        assert int(og["allowed"].sum()) == 1003 and int(og["valid"].sum()) == 128 and not any(k_.startswith(("att_", "kp_logsig")) for k_ in og), "the flag-off model must reproduce the E312 head"
    print("jends selftest ok; params", sum(p.numel() for p in JEnds().parameters()) / 1e6, "M")


if __name__ == "__main__":
    _selftest()

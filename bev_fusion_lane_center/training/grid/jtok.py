"""E357 (stage T0 of the token design E355, D105): the ground-truth nodes as tokens, ONE pair head for every edge of the graph (lane links and junction links).

    with lattice(2.5): ...                                   # `qtargets` on a 2.5 m lattice for the block (N, RES are module constants read at call time; restored after); check_lattice(2.5)
    g, allowed = pair_geometry(pos, dirv, valid)             # (B, N, N, 13) the pair features of `jends.JEnds.forward`, (B, N, N) the candidate gate; [b, i, j] = the link i -> j
    out = JTok(d=64)(P, pos, dirv, valid)                    # P = the pyramid dict (feat2, feat1, centre1) -> logit (B, N, N), scale (B, N, N, 2), allowed, chord
    loss, info = tok_loss(out, batch)                        # batch: y (B, N, N), ys (B, N, N, 2) of `data/lane_tok_data.collate`
    model = JTokModel(base="runs/e273_p_s0/model.pt")        # B's frozen pyramid (no gradient) + the head; forward_batch(model, batch); load_model(run, dev)

What the head is GIVEN per node: position and heading (the GT node table of `qtargets.node_table` on the lattice) and the frozen features of B's pyramid sampled at the position. NOT given: node flags, piece / section ids, the
order of the nodes (the head is a function of the SET: tested by a permutation), anything of the junction GT.
Token: MLP on [the 2 m features at the node (bilinear), the 1 m features as the mean of 9 bilinear samples at offsets of -1 / 0 / 1 m (a 3 x 3 patch: one tap would undersample the 1 m map, E347), position / 50, heading], then 2 set
blocks (`qlinks.SetBlock`, padding masked). Pair (i -> j): MLP on [token i, token j, the 13 geometric features, the lane heat of the pyramid (sigmoid of `centre1` channel 0) at 12 points of the chord-tangent Hermite curve of the pair:
mean and minimum] -> one logit; a second small MLP on the same input -> two handle scales 0.05 + 1.2 sigmoid (in chords, as the E303 scale head). All pairs are computed densely (N up to about 350).
The allowed gate (a candidate selection by geometry, named in the card): both headings point forward along the chord, turn at most 130 degrees (`qendnet.COS_TURN`), chord MIN_CHORD .. MAX_CHORD m, both nodes valid, i != j.
Loss (`tok_loss`, the exact form is in its docstring): BCE in two groups by chord (short <= 8 m, long > 8 m), each balanced on its own, + 1.0 x the dual softmax of `qendnet` + 0.5 x L1 on the handle scales of the long GT links.
No hard-negative mining and no averaging over the two directions of a pair (left out, named). Tests: test_jtok.py
"""
import contextlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import jmaps as JM
import qtargets as QT
from jends import _hermite, _sample
from qendnet import COS_TURN, _dual_softmax
from qlinks import SetBlock

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = 64
NG, NEV = 13, 2                                                             # geometric pair features, evidence features
MIN_CHORD, MAX_CHORD = 0.5, 100.0                                           # m: the chord range of the allowed gate
SHORT = QT.SHORT                                                            # 8 m: a link with a longer chord is a long link (`qtargets.node_table`)
HALF = QT.HALF


# ------------------------------------------------------------------------------------------------------------------------------ lattice
def _set_lattice(res):
    n = 2 * QT.HALF / res
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"cell size {res} m does not divide the {2 * QT.HALF} m window")
    QT.N, QT.RES = int(round(n)), float(res)


@contextlib.contextmanager
def lattice(res):
    """`qtargets` on a `res` m lattice inside the block (the mechanism of E346): N and RES are module constants that `cell_of` / `centre_of` read at call time, so every holder of the module sees them (qformat,
    qgraph_eval.LaneRef); a default bound at import does NOT (`grid_targets(n=N)`: pass n). Restored at the end, also after an exception. A data-loader worker enters the block itself (`lane_tok_data.frame_table`)."""
    old = QT.N, QT.RES
    _set_lattice(res)
    try:
        yield
    finally:
        QT.N, QT.RES = old


def check_lattice(res):
    """AssertionError unless `qtargets` is on the `res` m lattice in THIS process and nobody holds a second copy of the module (a copy would keep 2 m silently)"""
    import numpy as np
    n = int(round(2 * QT.HALF / res))
    assert (QT.N, QT.RES) == (n, float(res)), f"qtargets is on N = {QT.N}, RES = {QT.RES}, not on the {res} m lattice"
    mods = [(name, m) for name, m in list(sys.modules.items()) if m is not None]
    stale = [name for name, m in mods if getattr(getattr(m, "QT", None), "__name__", "").startswith("qtargets") and m.QT is not QT]
    copies = [name for name, m in mods if os.path.basename(getattr(m, "__file__", "") or "") == "qtargets.py" and m is not QT]
    assert not stale and not copies, f"a second copy of qtargets: held by {stale}, modules {copies}"
    xy = np.array([[-49.9, -49.9], [0.3, -0.7], [49.9, 49.9]])
    assert np.array_equal(QT.cell_of(xy), np.floor((xy + QT.HALF) / res).astype(int)), "cell_of does not use the lattice"


# ------------------------------------------------------------------------------------------------------------------------------ pairs
def pair_geometry(pos, dirv, valid=None, e=None, min_chord=MIN_CHORD, max_chord=MAX_CHORD):
    """pos, dirv (B, N, 2) metres / unit headings, valid (B, N) bool (None = all), e (B, N) existence probability of the nodes (None = 1: GT nodes)
    -> g (B, N, N, 13), allowed (B, N, N) bool; [b, i, j] = the link i -> j. The 13 features are those of `jends.JEnds.forward` (tested equal on the same pairs): cos and sin of the turn, chord / 20, the chord in the frame of
    heading i and of heading j (along, across; / 20), cos and sin of the chord direction against each heading, e_i, e_j. allowed: both headings forward along the chord, cos(turn) >= cos(130 degrees),
    min_chord <= chord <= max_chord, both valid, i != j."""
    B, N = pos.shape[:2]
    ch = pos[:, None, :, :] - pos[:, :, None, :]
    L = ch.norm(dim=-1)
    ta = dirv[:, :, None, :].expand_as(ch)
    tb = dirv[:, None, :, :].expand_as(ch)
    na = torch.stack([-ta[..., 1], ta[..., 0]], -1)
    nb = torch.stack([-tb[..., 1], tb[..., 0]], -1)
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
    cu = ch / L.clamp_min(1e-6)[..., None]
    cos_turn = dot(ta, tb)
    e = torch.ones_like(pos[..., 0]) if e is None else e
    valid = torch.ones_like(pos[..., 0], dtype=torch.bool) if valid is None else valid
    g = torch.stack([cos_turn, cross(ta, tb), L / 20.0, dot(ch, ta) / 20.0, dot(ch, na) / 20.0, dot(ch, tb) / 20.0, dot(ch, nb) / 20.0, dot(cu, ta), cross(ta, cu), dot(cu, tb), cross(tb, cu),
                     e[:, :, None].expand_as(L), e[:, None, :].expand_as(L)], -1)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)[None]
    allowed = (cos_turn >= COS_TURN) & (L >= min_chord) & (L <= max_chord) & (dot(ch, ta) > 0) & (dot(ch, tb) > 0) & valid[:, :, None] & valid[:, None, :] & ~eye
    return g, allowed


def lane_evidence(centre1, pos, dirv):
    """the lane heat along the candidate curve of every pair: centre1 (B, 12, 100, 100) the pyramid's centre output (channel 0 = lane-centre logit), -> (B, N, N, 2) mean and minimum of its sigmoid at 12 points of the
    chord-tangent Hermite curve i -> j (`jends._hermite`; the first two evidence features of `JEnds`)"""
    N = pos.shape[1]
    ex = lambda t, d: t.unsqueeze(d).expand(-1, N, N, -1)  # noqa: E731
    pts = _hermite(ex(pos, 2), ex(dirv, 2), ex(pos, 1), ex(dirv, 1))
    lane = _sample(centre1[:, :1].float().sigmoid(), pts)[..., 0]
    return torch.stack([lane.mean(-1), lane.min(-1)[0]], -1)


class JTok(nn.Module):
    """the token encoder + the pair head (module docstring); c2 / c1 = the channels of the pyramid's 2 m / 1 m features"""

    def __init__(self, c2=144, c1=48, d=D, hid=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(c2 + c1 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([SetBlock(d), SetBlock(d)])
        nin = 2 * d + NG + NEV
        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.scale = nn.Sequential(nn.Linear(nin, 32), nn.GELU(), nn.Linear(32, 2))
        o = torch.tensor([-1.0, 0.0, 1.0])
        self.register_buffer("patch", torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(9, 2), persistent=False)       # the 3 x 3 patch of 1 m cells around a node

    def tokens(self, P, pos, dirv, valid):
        f1 = _sample(P["feat1"], pos[:, :, None, :] + self.patch).mean(2)
        t = self.enc(torch.cat([_sample(P["feat2"], pos), f1, pos / HALF, dirv], -1))
        for blk in self.blocks:
            t = blk(t, ~valid)
        return t

    def forward(self, P, pos, dirv, valid):
        pos, dirv = pos.float(), dirv.float()
        N = pos.shape[1]
        t = self.tokens(P, pos, dirv, valid)
        g, allowed = pair_geometry(pos, dirv, valid)
        fcat = torch.cat([t[:, :, None].expand(-1, -1, N, -1), t[:, None].expand(-1, N, -1, -1), g, lane_evidence(P["centre1"], pos, dirv)], -1)
        return dict(logit=self.pair(fcat)[..., 0], scale=0.05 + 1.2 * self.scale(fcat).sigmoid(), allowed=allowed, chord=g[..., 2] * 20.0)


# ------------------------------------------------------------------------------------------------------------------------------ loss
def _balanced_bce(bce, y, m):
    """one group of pairs `m` (B, N, N): per frame 0.5 x (mean of `bce` over its positives + mean over its negatives); a class that a frame's group lacks adds nothing (the other keeps its 0.5);
    -> the mean over the frames that have a pair in the group (0 when none has)"""
    pos, neg = (m & (y > 0)).float(), (m & (y <= 0)).float()
    per = 0.5 * ((bce * pos).sum((1, 2)) / pos.sum((1, 2)).clamp_min(1.0) + (bce * neg).sum((1, 2)) / neg.sum((1, 2)).clamp_min(1.0))
    has = m.flatten(1).any(1)
    return per[has].mean() if has.any() else bce.sum() * 0.0


def tok_loss(out, batch, dual_w=1.0, scale_w=0.5):
    """loss = BCE_S + BCE_L + dual_w x dual softmax + scale_w x scale L1 -> (loss, info). With z the logits, y the GT link matrix (1 where i -> j is a GT link) and A the allowed pairs:
      BCE_G   G = S (A with chord <= 8 m) or L (A with chord > 8 m). Per frame f:  l_f = 1/2 [ mean over the positives of G_f of bce(z, 1) + mean over the negatives of G_f of bce(z, 0) ],
              i.e. inside a group the positives and the negatives of a frame carry the same total weight (a positive counts |G_f-| / |G_f+| times a negative); a frame whose group has only one class
              keeps that class at 1/2; BCE_G = the mean of l_f over the frames with at least one pair in G. (E273: the weight of the long positives decides; they are about 0.3 % of the pairs.)
      dual    `qendnet._dual_softmax(z, y, A)`: the candidates of a row (one source) and of a column (one target) compete with a 'none' entry of logit 0.
      scale   mean |scale - fitted scale| over the two handle scales of the GT links that are long, allowed and have a fit.
    GT links outside A are never trained (not forced into the gate): counted in info (`gt_outside`, of `gt_links`). A sigmoid at 0.5 is therefore NOT a calibrated probability of 1/2 (balanced classes)."""
    dev = out["logit"].device
    z, allowed, chord = out["logit"].float(), out["allowed"], out["chord"]
    y, ys = batch["y"].to(dev).float(), batch["ys"].to(dev).float()
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    l_s, l_l = _balanced_bce(bce, y, allowed & (chord <= SHORT)), _balanced_bce(bce, y, allowed & (chord > SHORT))
    dual = _dual_softmax(z, y, allowed) if allowed.any() else z.sum() * 0.0
    loss = l_s + l_l + dual_w * dual
    info = dict(bce_s=float(l_s), bce_l=float(l_l), dual=float(dual), allowed=float(allowed.sum()), pos_pairs=float((y * allowed).sum()), gt_links=float(y.sum()), gt_outside=float((y * ~allowed).sum()))
    m = (y > 0) & allowed & torch.isfinite(ys[..., 0])
    if m.any():
        ls = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * ls
        info["scale_l1"] = float(ls)
    return loss, info


# ------------------------------------------------------------------------------------------------------------------------------ model
class JTokModel(nn.Module):
    """B's pyramid (`runs/e273_p_s<k>/model.pt`, tensors `pyr.*`, loaded by `jmaps.JModel.load_base`; frozen, run without gradient) + the token head"""

    def __init__(self, d=D, base=None):
        super().__init__()
        jm = JM.JModel(width=48, f05=32)
        if base is not None:
            n, tot = jm.load_base(base)
            assert n == tot, f"{base}: {n} of {tot} pyramid tensors fit"
        self.pyr, self.tok = jm.pyr, JTok(c2=3 * 48, c1=48, d=d)
        for p in self.pyr.parameters():
            p.requires_grad_(False)

    def forward(self, x, pos, dirv, valid):
        with torch.no_grad():
            P = self.pyr(x)
        return self.tok(P, pos, dirv, valid)


def forward_batch(model, b):
    """a batch of `lane_tok_data.collate` through the model: the ONE place that says what of a batch the head reads (x, pos, dirv, valid; no flags, no piece ids)"""
    dev = next(model.parameters()).device
    return model(b["x"].to(dev, non_blocking=True).float(), b["pos"].to(dev), b["dirv"].to(dev), b["valid"].to(dev))


def load_model(run, dev, runs=None):
    """-> (the model of runs/<run>/model.pt in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or os.path.join(ROOT, "runs"), run, "model.pt"), map_location="cpu")
    m = JTokModel(d=ck["args"]["d"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]

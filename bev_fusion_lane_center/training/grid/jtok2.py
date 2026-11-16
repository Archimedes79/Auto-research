"""E359 (the question left open by E357): the GT nodes as tokens, a node-type head and a per-source softmax over the candidate successors with a "none" option.

    model = JTok2Model(base="runs/e273_p_s0/model.pt")        # B's frozen pyramid + `JTok2` = the E357 token head + `types`; `forward_batch(model, batch)` and `load_model(run, dev)` as in `jtok`
    out = model_out                                           # logit (B, N, N), scale, allowed, chord (all as `jtok.JTok`) + types (B, N, 3) = the logits (long_out, long_in, short_out) of each node
    loss, info = tok2_loss(out, batch)                        # the exact likelihood below
    pr = probs(out)                                           # s (B, N, N) the conditional softmax of the class of each pair, tp (B, N, 3) the type probabilities, longm (B, N, N)

Everything of E357 stays (nodes, features, token encoder, two set blocks, no flags / piece ids / order, the allowed gate, the pair logit z and the two handle scales: `jtok.JTok` is the parent class). New: `types` = ONE Linear(d, 3) on
the token after the set blocks: long_out (the node is the SOURCE of a GT link with chord > 8 m), long_in (the DESTINATION of such a link), short_out (the SOURCE of a GT link with chord <= 8 m). GT links outside the allowed gate
are masked out of every target (and counted: `info["gt_outside"]`).

Likelihood (`tok2_loss`). Per class c (short: chord <= 8 m; long: chord > 8 m) each SOURCE i with at least one candidate of the class (allowed pair, class chord; a node with none is masked from the class entirely) distributes
one unit of probability over "none" and its candidates:  p(none) = 1 - sigmoid(a_i),  p(j) = sigmoid(a_i) x s_ij,  s_ij = softmax over the candidates j of z'_ij (float32, log space, masked entries -1e4 as in `qendnet`),
a_i = short_out(i) or long_out(i); z'_ij = z_ij for the short class and z_ij + logsigmoid(b_j) for the long class (b_j = long_in(j): the destination's own evidence enters the RANKING of the long candidates).
With G(i) = the in-gate GT successors of i in the class (multi-positive):
    G(i) empty:      loss_i = -log(1 - sigmoid(a_i))
    G(i) non-empty:  loss_i = -log( sum_{j in G} p(j) ) = -[ logsigmoid(a_i) + logsumexp_{G} z'_ij - logsumexp_{cand} z'_ij ]
The type BCE of the source-side logits IS the none part of this likelihood (the first term of each line is exactly the plain BCE of a_i against its target: 1 when G is non-empty): there is NO separate BCE on long_out /
short_out and no 'none' entry inside the softmax (nothing is counted twice). The pair logit gets gradient only from the sources that have a GT link (the others reach the network through a_i only).
Destination side: long_in is trained by a plain BCE (weight 1) on the nodes that are a candidate destination of at least one long candidate (target: the node is the destination of an in-gate GT long link); it also receives
the ranking gradient through z'. No destination-side softmax. Plus 0.5 x L1 on the two handle scales of the long in-gate GT links (as `jtok.tok_loss`).
loss = L_short + L_long + BCE(long_in) + 0.5 x scale L1. Normalisation: L_c = per frame the mean of loss_i over the sources of the class (rows above), then the mean over the frames that have such a source; BCE(long_in) = per frame
the mean over its nodes that are a long candidate destination, then the mean over the frames. Tests: test_jtok2.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402

SHORT = JT.SHORT
NEG = -1e4                                                                  # the fill of a masked logit (`qendnet._dual_softmax`)
TYPES = ("long_out", "long_in", "short_out")                                # the columns of `types`


class JTok2(JT.JTok):
    """`jtok.JTok` + `types`: Linear(d, 3) on the token after the set blocks (copy of the parent's forward, which does not expose the tokens)"""

    def __init__(self, c2=144, c1=48, d=JT.D, hid=64):
        super().__init__(c2, c1, d, hid)
        self.types = nn.Linear(d, 3)

    def forward(self, P, pos, dirv, valid):
        pos, dirv = pos.float(), dirv.float()
        N = pos.shape[1]
        t = self.tokens(P, pos, dirv, valid)
        g, allowed = JT.pair_geometry(pos, dirv, valid)
        fcat = torch.cat([t[:, :, None].expand(-1, -1, N, -1), t[:, None].expand(-1, N, -1, -1), g, JT.lane_evidence(P["centre1"], pos, dirv)], -1)
        return dict(logit=self.pair(fcat)[..., 0], scale=0.05 + 1.2 * self.scale(fcat).sigmoid(), allowed=allowed, chord=g[..., 2] * 20.0, types=self.types(t))


class JTok2Model(JT.JTokModel):
    """B's frozen pyramid + `JTok2` (`forward_batch` and the state layout `tok.*` as in `jtok`)"""

    def __init__(self, d=JT.D, base=None):
        super().__init__(d=d, base=base)
        self.tok = JTok2(c2=3 * 48, c1=48, d=d)


forward_batch = JT.forward_batch


def load_model(run, dev, runs=None):
    """-> (the model of runs/<run>/model.pt in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or os.path.join(JT.ROOT, "runs"), run, "model.pt"), map_location="cpu")
    m = JTok2Model(d=ck["args"]["d"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


# ------------------------------------------------------------------------------------------------------------------------------ classes and targets
def class_masks(allowed, chord):
    """(short, long) candidate masks (B, N, N): the allowed pairs by chord (<= 8 m / > 8 m)"""
    return allowed & (chord <= SHORT), allowed & (chord > SHORT)


def gt_classes(y, allowed, chord):
    """(GS, GL): the in-gate GT links of the short / long class (B, N, N) bool; the targets of every node type are their row / column any()"""
    S, L = class_masks(allowed, chord)
    yb = y > 0
    return S & yb, L & yb


def rank_logits(out):
    """z' (B, N, N) float32: the pair logit, plus logsigmoid(long_in of the destination) on the pairs with a chord above 8 m"""
    z, b = out["logit"].float(), out["types"].float()[..., 1]
    return z + torch.where(out["chord"] > SHORT, F.logsigmoid(b)[:, None, :], torch.zeros_like(z))


def _lse(zr, m):
    """logsumexp over the destinations of the entries in `m` (B, N, N bool); a row without an entry gives the finite -1e4 + log N (its loss is masked by the caller)"""
    return torch.logsumexp(torch.where(m, zr, torch.full_like(zr, NEG)), 2)


def probs(out):
    """the factorisation as probabilities: s (B, N, N) = the softmax of z' over the candidates of the pair's class (0 outside the gate; sums to 1 over the candidates of each class of a source), tp (B, N, 3) =
    sigmoid(types) = (long_out, long_in, short_out), longm (B, N, N) = chord above 8 m. The probability of the edge i -> j is tp[i, class] * s[i, j]"""
    zr = rank_logits(out)
    S, L = class_masks(out["allowed"], out["chord"])
    s = sum(torch.exp(torch.where(m, zr, torch.full_like(zr, NEG)) - _lse(zr, m)[..., None]) * m for m in (S, L))
    return dict(s=s, tp=out["types"].float().sigmoid(), longm=out["chord"] > SHORT)


# ------------------------------------------------------------------------------------------------------------------------------ loss
def _source_nll(zr, cand, gt, a, valid):
    """the per-source loss of one class (module docstring) -> (nll (B, N), rows (B, N) bool: the sources of the class)"""
    has = gt.any(2)
    nll = torch.where(has, -(F.logsigmoid(a) + _lse(zr, gt) - _lse(zr, cand)), -F.logsigmoid(-a))
    return nll, cand.any(2) & valid


def _frame_mean(x, m):
    """per frame the mean of x over the entries of m, then the mean over the frames that have one (0 when none)"""
    per = (x * m).sum(1) / m.sum(1).clamp_min(1)
    has = m.any(1)
    return per[has].mean() if has.any() else x.sum() * 0.0


def tok2_loss(out, batch, scale_w=0.5):
    """the exact likelihood of the module docstring -> (loss, info); batch of `data/lane_tok_data.collate` (y, ys, valid). info: nll_s, nll_l, bce_in, scale_l1 and the counts allowed, pos_pairs (in-gate GT links),
    gt_links, gt_outside"""
    dev = out["logit"].device
    allowed, chord, ty = out["allowed"], out["chord"], out["types"].float()
    y, ys, valid = batch["y"].to(dev).float(), batch["ys"].to(dev).float(), batch["valid"].to(dev)
    S, L = class_masks(allowed, chord)
    GS, GL = gt_classes(y, allowed, chord)
    zr = rank_logits(out)
    nll_s = _frame_mean(*_source_nll(zr, S, GS, ty[..., 2], valid))
    nll_l = _frame_mean(*_source_nll(zr, L, GL, ty[..., 0], valid))
    cols = L.any(1) & valid                                                  # the nodes that are a candidate destination of a long link
    bce_in = _frame_mean(F.binary_cross_entropy_with_logits(ty[..., 1], GL.any(1).float(), reduction="none"), cols)
    loss = nll_s + nll_l + bce_in
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), bce_in=float(bce_in), allowed=float(allowed.sum()), pos_pairs=float((GS | GL).sum()), gt_links=float(y.sum()), gt_outside=float((y * ~allowed).sum()))
    m = GL & torch.isfinite(ys[..., 0])
    if m.any():
        ls = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * ls
        info["scale_l1"] = float(ls)
    return loss, info

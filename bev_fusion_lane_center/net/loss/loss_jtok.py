"""Losses of the ground-truth token pair head: the loss side of net/model/jtok.py (D152; the functions are unchanged, `jtok` holds the network)."""
import torch
import torch.nn.functional as F

from jtok import SHORT  # noqa: E402,I001
from qendnet import _dual_softmax


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

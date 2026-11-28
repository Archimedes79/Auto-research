"""Losses of the token head (the wiring likelihood): the loss side of net/model/jtoken.py (D152; `jtoken` holds the network)."""
import torch
import torch.nn.functional as F

from lane_center.net.model.jtoken import NEG, SHORT  # noqa: E402,I001


def _frame_mean(x, m, w=None):
    """per frame the mean of x (B, N) over the entries of m, then the mean over the frames that have one (0 when none); w (B, N): the weight of every entry, the mean is then sum(w x) / sum(w) as torch's class-weighted cross-entropy
    (x is the already weighted term); w None = the plain mean"""
    per = (x * m).sum(1) / (m.sum(1).clamp_min(1) if w is None else (w * m).sum(1).clamp_min(1e-12))
    has = m.any(1)
    return per[has].mean() if has.any() else x.sum() * 0.0


# ------------------------------------------------------------------------------------------------------------------------------ loss
def _source_nll(lp, G, none_weight):
    """per source and class: lp (B, 2, K, K + 1) the log-probabilities of the per-source softmax (the none slot last), G (B, 2, K, K) the in-class GT successor groups -> (B, 2, K). G(i) non-empty: the likelihood of
    its GT successors, -log sum_{G(i)} p; G(i) empty: `none_weight` x -log p(none|i) (the sources with a GT successor are not touched by the weight)"""
    K = G.shape[-1]
    g = lp[..., :K].masked_fill(~G, NEG)
    return torch.where(G.any(-1), -torch.logsumexp(g, -1), -lp[..., K] * none_weight)


def source_rows(out):
    """(B, 2, K) bool: the sources of a class, the valid tokens with a candidate of it (the rows of the likelihood)"""
    return out["cand"].any(-1) & out["valid"][:, None]


def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0):
    """the likelihood of the module docstring -> (loss, info); batch: y (B, K, K), ylong (B, K, K), yscale (B, K, K, 2), ytype (B, K). info: nll_s, nll_l, ce1, ce3, scale_l1 and the counts gt_links, pos_pairs (in-gate GT
    links), gt_outside (links with no gate), gt_class (in the gate, but `ylong` disagrees with the token chord: a statistic, the class of a link is the token chord's). The long scale target is the fitted yscale
    of the long-class links (finite where the cache fitted one). none_weight W (default 1): multiplies (i) the loss of the sources whose G(i) is empty (their target is the none slot), in both classes, and (ii) the type
    cross-entropy of the tokens whose type is none (class 0 weight W, the six others 1, normalised by the sum of the weights of the valid tokens of the frame, as torch's weighted mean).
    scale_w = 0: no scale term, `yscale` is not read"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = y[:, None] & cand                                                  # (B, 2, K, K): the in-gate GT successors of each class (the class of a pair = its token chord)
    yt = batch["ytype"].to(dev).long()
    nll = _source_nll(lp, G, none_weight)
    rows = source_rows(out)
    nll_s, nll_l = (_frame_mean(nll[:, c], rows[:, c]) for c in range(2))
    wt = torch.where(yt == 0, none_weight, 1.0) if none_weight != 1.0 else None
    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), ce1=float(ce1), ce3=float(ce3), gt_links=float(y.sum()), pos_pairs=float(G.sum()), gt_outside=float((y & ~cand.any(1)).sum()),
                gt_class=float((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    if not scale_w:
        return loss, info
    ys = batch["yscale"].to(dev).float()
    m = G[:, 1] & torch.isfinite(ys[..., 0])
    if m.any():
        l1 = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * l1
        info["scale_l1"] = float(l1)
    return loss, info

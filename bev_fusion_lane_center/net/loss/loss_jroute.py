"""Losses of the route association head: the loss side of net/model/jroute.py (D152; the functions are unchanged, `jroute` holds the network)."""
import torch
import torch.nn.functional as F

from jroute import LABELS  # noqa: E402,I001


# ------------------------------------------------------------------------------------------------------------------------------ the loss and the label frequencies
def route_loss(out, batch, pos_weight=(1.0, 1.0)):
    """BCE of the road and the lane logit on the tokens with a label != -1 (the label of a token that is assigned to no GT node, of a padding token and of a frame without a route is -1), a positive token weighted by pos_weight = (road, lane)
    (torch's `pos_weight`: the mean over the labelled tokens of w y (-log p) + (1 - y) (-log (1 - p))); loss = bce_road + bce_lane -> (loss, info: bce_road, bce_lane, acc_road, acc_lane (0 without a labelled token), n_road, n_lane = the labelled
    tokens, pos_road, pos_lane = the positives among them). A batch without a labelled token has the loss 0 with a graph (zero gradient)"""
    loss, info = out["road"].sum() * 0.0, {}
    for name, key, w in zip(("road", "lane"), LABELS, pos_weight):
        lo, y = out[name], batch[key].to(out[name].device)
        m = y >= 0
        n = m.sum()
        t = (y == 1).float()
        bce = F.binary_cross_entropy_with_logits(lo, t, reduction="none") * torch.where(y == 1, float(w), 1.0)
        term = (bce * m).sum() / n.clamp_min(1)
        loss = loss + term
        info.update({f"bce_{name}": float(term), f"acc_{name}": float((((lo > 0).float() == t) & m).sum() / n.clamp_min(1)), f"n_{name}": float(n), f"pos_{name}": float(((y == 1) & m).sum())})
    return loss, info

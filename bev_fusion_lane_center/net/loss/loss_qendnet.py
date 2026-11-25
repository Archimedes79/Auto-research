"""Losses of the junction end head: the loss side of net/model/qendnet.py (D152; the functions are unchanged, `qendnet` holds the network)."""
import torch
import torch.nn.functional as F

from qendnet import OFF, _dual_softmax, cell_index, match_keypoints  # noqa: E402,I001
from qnodes import cell_centres


SIGMA = 2.0                                                                 # m: the Gaussian of the tolerant target


NEAR = 3.0                                                                  # m: offset / heading are trained within this distance of a GT end


def _focal(logit, t, n):
    p = logit.sigmoid()
    pos = t >= 1.0
    lp = -((1 - p) ** 2 * F.logsigmoid(logit))[pos].sum()
    ln = -(((1 - t) ** 4) * p ** 2 * F.logsigmoid(-logit))[~pos].sum()
    return (lp + ln) / n


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
        t = t.scatter_reduce(1, cell, m.to(t.dtype), "amax", include_self=True)       # the cell that holds an end is the peak (target 1); padded ends add 0
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

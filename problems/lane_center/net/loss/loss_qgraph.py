"""Losses of the lane head (pixel, node, link and assignment stages): the loss side of net/model/qgraph.py (D152; `qgraph` holds the network). The loss switches of E274 / E276 / E282 and the end head's loss (E284) are in the tag v1."""
import torch
import torch.nn.functional as F

from lane_center.net.model.qgraph import PHASE_FROZEN  # noqa: E402,I001
from lane_center.net.loss import losses as GL
from lane_center.net.model import qnodes as QN
from lane_center.net.loss import loss_sections as LSE  # noqa: E402


PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign", 9: "assign"}


W = dict(pix=1.0, node=1.0, link=1.0, sec=1.0, asg=1.0)


def focal_bce(logit, y, alpha=0.5, gamma=2.0, norm=None):
    p = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    l = ce * (1 - pt) ** gamma * (alpha * y + (1 - alpha) * (1 - y))
    return l.sum() / (norm if norm is not None else y.sum().clamp_min(1.0))


def dice(logit, y, eps=1.0):
    p = logit.sigmoid()
    inter = (p * y).sum((-2, -1))
    return (1 - (2 * inter + eps) / (p.sum((-2, -1)) + y.sum((-2, -1)) + eps)).mean()


def pixel_loss(P, batch):
    dev = P["centre1"].device
    info = {}
    l1, i1 = GL.grid_loss(P["centre1"][:, :6], dict(g05=batch["g05"].to(dev), t05=batch["t05"].to(dev), m05=batch["m05"].to(dev)))
    info["centre"] = float(l1)
    ev, evt = P["centre1"][:, 6:12].float(), batch["events1"].to(dev).float()
    l_ev = F.binary_cross_entropy_with_logits(ev, evt, pos_weight=torch.tensor(20.0, device=dev)) + dice(ev, evt)
    info["events"] = float(l_ev)
    lm, ld = P["lines05"][:, :3].float(), P["lines05"][:, 3:].float()
    ym, yd = batch["lines_mask"].to(dev).float(), batch["lines_dist"].to(dev).float()
    l_ln = F.binary_cross_entropy_with_logits(lm, ym, pos_weight=torch.tensor(8.0, device=dev)) + dice(lm, ym) + (2.0 * ld.sigmoid() - yd).abs().mean() * 0.5
    info["lines"] = float(l_ln)
    rd, yr = P["road2"].float(), batch["road2"].to(dev).float()
    l_rd = F.binary_cross_entropy_with_logits(rd, yr) + dice(rd, yr)
    info["road"] = float(l_rd)
    return l1 + l_ev + l_ln + l_rd, info


def node_loss(no, batch):
    dev = no["e"].device
    ex = batch["exists"].to(dev).float()
    m = ex > 0.5
    npos = ex.sum().clamp_min(1.0)
    info = {}
    l_e = focal_bce(no["e"].float(), ex, norm=npos)
    fl = batch["flags"].to(dev).float()
    l_f = (F.binary_cross_entropy_with_logits(no["flags"].float(), fl, reduction="none", pos_weight=torch.tensor(6.0, device=dev)) * m[:, None]).sum() / npos
    off_t = batch["offset"].to(dev).float()
    l_o = ((no["off"].float() - off_t).abs().sum(1) * m).sum() / npos
    dr = QN.unit(no["dir_raw"].float())
    l_d = ((1 - (dr * batch["dirv"].to(dev).float()).sum(1)) * m).sum() / npos
    l_s = (F.binary_cross_entropy_with_logits(no["s"].float(), batch["sstart"].to(dev).float(), reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * m).sum() / npos
    l_u = sum((F.binary_cross_entropy_with_logits(no[k].float(), batch[k].to(dev).float(), reduction="none", pos_weight=torch.tensor(8.0, device=dev)) * m).sum() / npos for k in ("uout", "uin"))
    info.update(e=float(l_e), flags=float(l_f), off=float(l_o), dir=float(l_d), s=float(l_s), u=float(l_u))
    return l_e + 0.5 * l_f + l_o + 0.5 * l_d + 0.5 * l_s + 2.0 * l_u, info


def link_loss(L, batch):
    dev = L["logit"].device
    y = L["label"]
    npos = y.sum().clamp_min(1.0)
    info = {}
    lg = L["is_long"]
    sh = ~lg
    l_b = focal_bce(L["logit"].float()[sh], y[sh], norm=y[sh].sum().clamp_min(1.0))               # short links: focal BCE over all candidate pairs
    l_long = torch.zeros((), device=dev)
    if lg.any():                                                                                  # long links: their positives are ~5 % of all positives and ~0.3 % of their own candidates: a balanced BCE of their own
        yl = y[lg]
        npl, nnl = yl.sum(), (1 - yl).sum()
        wgt = torch.where(yl > 0.5, (nnl / npl.clamp_min(1.0)).clamp(1.0, 15.0), torch.ones_like(yl))
        l_long = (F.binary_cross_entropy_with_logits(L["logit"].float()[lg], yl, reduction="none") * wgt).sum() / (wgt.sum() + 1.0)
    # degree: the expected number of outgoing links of every active node against the GT degree of its cell
    B, K = L["valid"].shape
    p = L["logit"].float().sigmoid()
    deg = torch.zeros(B, K, device=dev).index_put((L["pb"], L["pa"]), p, accumulate=True)
    lk = batch["lk"].to(dev)
    okl = lk[..., 0] >= 0
    gdeg = torch.zeros(B, QN.N * QN.N, device=dev)
    bb = torch.arange(B, device=dev)[:, None].expand_as(okl)
    gdeg.index_put_((bb[okl], lk[..., 0][okl] * QN.N + lk[..., 1][okl]), torch.ones(int(okl.sum()), device=dev), accumulate=True)
    gd = gdeg.gather(1, L["act"])
    l_deg = ((deg - gd).abs() * L["valid"]).sum() / L["valid"].sum().clamp_min(1.0)
    # handle scales of the positive long links against the fitted ones
    l_sc = torch.zeros((), device=dev)
    pos = (y > 0.5) & L["is_long"]
    if pos.any() and "lk_scale" in batch:
        sc = batch["lk_scale"].to(dev)
        key_g = (bb[okl] * QN.N * QN.N + lk[..., 0][okl] * QN.N + lk[..., 1][okl]) * (QN.N * QN.N) + lk[..., 2][okl] * QN.N + lk[..., 3][okl]
        order = key_g.argsort()
        kg = key_g[order]
        sg = sc[okl][order]
        key_p = (L["pb"] * QN.N * QN.N + L["act"][L["pb"], L["pa"]]) * (QN.N * QN.N) + L["act"][L["pb"], L["pc"]]
        j = torch.searchsorted(kg, key_p[pos]).clamp_max(len(kg) - 1)
        hit = (kg[j] == key_p[pos]) & ~torch.isnan(sg[j, 0])
        if hit.any():
            tgt = sg[j][hit]
            l_sc = ((L["sa"][pos][hit] - tgt[:, 0]).abs() + (L["sb"][pos][hit] - tgt[:, 1]).abs()).mean()
    info.update(link=float(l_b), long=float(l_long), deg=float(l_deg), scale=float(l_sc), pos=int(npos), pairs=len(y), miss=int(L.get("gt_missing", 0)))
    return l_b + 2.0 * l_long + 2.0 * l_sc, info                                  # the degree term is logged, not trained: with one-to-many labels a node has several true successors


def assign_loss(A, L, batch):
    dev = A["P_piece"].device
    Pp = A["P_piece"]
    B, Ka, K1 = Pp.shape
    K = K1 - 1
    piece = batch["piece"].to(dev).flatten(1).gather(1, L["act"])                              # GT piece index of every active node (-1 none)
    valid = L["valid"] & (piece >= 0)
    pm = batch["piece_map"].to(dev)                                                            # (B, Pm, K)
    P_ = pm.shape[1]
    tgt = pm.gather(1, piece.clamp_min(0)[..., None].expand(-1, -1, K)) & valid[..., None]    # (B, Ka, K) multi-hot of the map pieces of the node's GT piece
    has = tgt.any(-1)
    mass = (Pp[..., :K] * tgt.float()).sum(-1) + torch.where(has, torch.zeros_like(Pp[..., K]), Pp[..., K])
    l_a = (-torch.log(mass.clamp_min(1e-6)) * valid).sum() / valid.sum().clamp_min(1.0)
    fp = L["valid"] & (piece < 0)
    l_fp = (-torch.log(Pp[..., K].clamp_min(1e-6)) * fp).sum() / fp.sum().clamp_min(1.0)
    # road level
    pr = batch["piece_road"].to(dev).gather(1, piece.clamp_min(0))                              # (B, Ka)
    Pr = A["P_road"]
    R = Pr.shape[-1] - 1
    rm = F.one_hot(pr.clamp(0, R - 1), R).float() * (pr >= 0)[..., None]
    rmass = (Pr[..., :R] * rm).sum(-1) + torch.where(pr >= 0, torch.zeros_like(Pr[..., R]), Pr[..., R])
    l_r = (-torch.log(rmass.clamp_min(1e-6)) * valid).sum() / valid.sum().clamp_min(1.0)
    # structure: GT-linked nodes must land on map pieces joined by the map's successor relation
    pos = L["label"] > 0.5
    l_st = torch.zeros((), device=dev)
    if pos.any():
        pb, pa, pc = L["pb"][pos], L["pa"][pos], L["pc"][pos]
        Pa, Pc = Pp[pb, pa], Pp[pb, pc]
        C = torch.eye(K1, device=dev)[None].repeat(len(pb), 1, 1)
        A1 = batch["map_adj"].to(dev)[pb, 0].float()
        C[:, :K, :K] += A1 + A1 @ A1                                                           # a long link crosses a suppressed connector piece: two successor steps
        C.clamp_(max=1.0)
        s = torch.einsum("pi,pij,pj->p", Pa, C, Pc)
        l_st = -torch.log(s.clamp_min(1e-6)).mean()
    info = dict(asg=float(l_a), fp=float(l_fp), road=float(l_r), struct=float(l_st))
    with torch.no_grad():
        info["acc"] = float((((Pp[..., :K] * tgt.float()).sum(-1) > 0.5) & has & valid).sum() / (has & valid).sum().clamp_min(1))
    return l_a + 0.3 * l_fp + 0.5 * l_r + 0.5 * l_st, info


def qgraph_loss(model, batch, phase):
    out = model(batch, upto=PHASE_UPTO[phase], frozen=PHASE_FROZEN[phase])
    loss, info = 0.0, {}
    if phase in (2, 7):
        l, i = pixel_loss(out["P"], batch)
        loss, info = loss + W["pix"] * l, {**info, **{"px_" + k: v for k, v in i.items()}}
    if phase in (3, 5, 7, 9):
        l, i = node_loss(out["no"], batch)
        loss, info = loss + W["node"] * l, {**info, **{"n_" + k: v for k, v in i.items()}}
    if phase in (4, 7, 9):
        l, i = link_loss(out["L"], batch)
        loss, info = loss + W["link"] * l, {**info, **{"l_" + k: v for k, v in i.items()}}
    if phase in (5, 7, 9):
        ce, pl = LSE.section_loss(out["M"], dict(piece=batch["piece"].to(out["L"]["act"].device), sstart=batch["sstart"].to(out["L"]["act"].device)), out["L"])
        loss, info = loss + W["sec"] * (ce + pl), {**info, "s_ce": float(ce), "s_pair": float(pl)}
    if phase in (6, 7, 9):
        l, i = assign_loss(out["A"], out["L"], batch)
        loss, info = loss + W["asg"] * l, {**info, **{"a_" + k: v for k, v in i.items()}}
    return loss, info

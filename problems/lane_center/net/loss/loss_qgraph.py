"""Losses of the lane head (pixel, node, link and assignment stages): the loss side of net/model/qgraph.py (D152; the functions are unchanged, `qgraph` holds the network)."""
import math
import torch
import torch.nn.functional as F

from qgraph import PHASE_FROZEN  # noqa: E402,I001
import losses as GL
import loss_qendnet as LQEN  # noqa: E402
import qnodes as QN
import loss_sections as LSE  # noqa: E402


PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign", 9: "assign"}


W = dict(pix=1.0, node=1.0, link=1.0, sec=1.0, asg=1.0)


# E274 switches (the defaults are the E273 loss): ev_w = weight of the 1 m event heat-map loss in the pixel stage; node_w_evt / node_w_inter = extra weight of the POSITIVES of the existence loss
# at event cells / at cells inside the (grown) intersection mask (weight 1 + w_evt * event + w_inter * inter); long_match = weight of the dual-softmax loss over the long-link candidates (E274)
# E276 switches (defaults = the E273 loss; they need the l4 maps tier / vis / ztier in the batch): tier_w = (w1, w2, w3) factor of the POSITIVE terms of the node loss (existence, flags, offset, direction,
# section start, long-link endpoints) at a GT node of tier 1 / 2 / 3; vis_w = extra factor for GT nodes that no camera sees (0 = not demanded); neg_w = (z1, z2, z3, z0) factor of the NEGATIVES of the
# existence loss by the zone of the cell (the lowest tier of the GT lane points near it, z0 = no lane: the cost of a false node near the own lane); link_tier_w = (w1, w2, w3) factor of the link losses
# by the tier of the source node (and vis_w for invisible ones)
# E282 switches (defaults = the E273 loss): node_loss = "zone" is the zone-matching existence loss on the sparse target (`zone_targets`; the batch needs the maps of `qzones.zone_maps`), zone_bias = the
# logit bonus of the cell that holds the sparse GT node when the winner of a zone is chosen (a tie-break only: 0.05)
CFG = dict(ev_w=1.0, node_w_evt=0.0, node_w_inter=0.0, long_match=0.0, tier_w=None, vis_w=1.0, neg_w=None, link_tier_w=None, node_loss="e273", zone_bias=0.05)


def focal_bce(logit, y, alpha=0.5, gamma=2.0, norm=None, pos_w=None, neg_w=None, w=None):
    p = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a_pos = alpha if pos_w is None else alpha * pos_w                                   # pos_w: a per-element weight of the positives (E274)
    a_neg = (1 - alpha) if neg_w is None else (1 - alpha) * neg_w                       # neg_w: a per-element weight of the negatives (E276)
    l = ce * (1 - pt) ** gamma * (a_pos * y + a_neg * (1 - y))
    if w is not None:                                                                   # w: a per-element weight of positives and negatives alike (E276, the links)
        l = l * w
    return l.sum() / (norm if norm is not None else y.sum().clamp_min(1.0))


def tier_lut(values, dev, first=1.0):
    """a lookup table indexed by a tier 0 / 1 / 2 / 3: tier 0 (no label) -> `first`, tiers 1 - 3 -> `values`"""
    return torch.tensor([first, *[float(v) for v in values]], device=dev)


def node_pos_weight(batch, dev):
    """E276: the (B, 50, 50) factor of the positive terms of the node loss from the tier and visibility maps of the batch (tier_w, vis_w); None when both are at their defaults"""
    if CFG["tier_w"] is None and CFG["vis_w"] == 1.0:
        return None
    tw = torch.ones(batch["exists"].shape, device=dev)
    if CFG["tier_w"] is not None:
        tw = tier_lut(CFG["tier_w"], dev)[batch["tier"].to(dev).long().clamp(0, 3)]
    if CFG["vis_w"] != 1.0:
        vis = batch["vis"].to(dev).float()
        tw = tw * (vis + (1.0 - vis) * CFG["vis_w"])
    return tw


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
    return l1 + CFG["ev_w"] * l_ev + l_ln + l_rd, info


def zone_targets(e_logit, batch, bias):
    """E282 zone-matching: per zone of the sparse target (batch `zone`, (B, 50, 50) int, -1 = none) the cell with the largest `e_logit + bias * zone_gt` is the positive (an index choice WITHOUT gradient, like the
    index choices of `qlinks.py`); exact ties go to the lowest flat cell index. -> (exists_m (B, 50, 50) float with one 1 per zone, info dict of tensors: n_zones, win_off_gt = the share of FREE zones whose winner
    is not the cell of the sparse GT node)"""
    dev = e_logit.device
    zone = batch["zone"].to(dev).long()
    gt = batch["zone_gt"].to(dev).float()
    B, H, W = zone.shape
    flat = zone.reshape(-1)
    valid = flat >= 0
    ex = torch.zeros(B * H * W, device=dev)
    if not bool(valid.any()):
        return ex.view(B, H, W), dict(n_zones=torch.zeros((), device=dev), win_off_gt=torch.zeros((), device=dev))
    zmax = int(flat.max().item()) + 1
    seg = (torch.arange(B, device=dev)[:, None, None] * zmax + zone).reshape(-1)[valid]
    score = (e_logit.detach().float() + bias * gt).reshape(-1)[valid]
    idx = torch.arange(B * H * W, device=dev)[valid]
    nseg = B * zmax
    mx = torch.full((nseg,), float("-inf"), device=dev).scatter_reduce(0, seg, score, "amax", include_self=True)
    big = B * H * W
    cand = torch.where(score >= mx[seg], idx, torch.full_like(idx, big))
    win = torch.full((nseg,), big, device=dev, dtype=torch.long).scatter_reduce(0, seg, cand, "amin", include_self=True)
    win = win[win < big]
    ex[win] = 1.0
    free = batch["zone_kind"].to(dev).reshape(-1)[win] == 1
    on_gt = gt.reshape(-1)[win] > 0.5
    off_gt = (~on_gt[free]).float().mean() if bool(free.any()) else torch.zeros((), device=dev)
    return ex.view(B, H, W), dict(n_zones=torch.tensor(float(win.numel()), device=dev), win_off_gt=off_gt)


def node_loss(no, batch):
    dev = no["e"].device
    zinfo = None
    if CFG["node_loss"] == "zone":                                                             # E282: the winners of the zones are the positives; offset and direction are those of the winner's DENSE GT node
        assert CFG["tier_w"] is None and CFG["vis_w"] == 1.0 and CFG["neg_w"] is None, "the zone loss runs without the E276 tier switches"
        ex, zinfo = zone_targets(no["e"], batch, CFG["zone_bias"])
        off_key, dir_key = "offset_dense", "dirv_dense"
    else:
        ex = batch["exists"].to(dev).float()
        off_key, dir_key = "offset", "dirv"
    m = ex > 0.5
    npos = ex.sum().clamp_min(1.0)
    info = {}
    wmap = None
    if CFG["node_w_evt"] or CFG["node_w_inter"]:                                           # E274: recall where the nodes are weak (event cells, intersections)
        evt = (batch["flags"].to(dev).float().max(1)[0] > 0.5).float()
        inter = (batch["road2"].to(dev).float()[:, 1] > 0.5).float()
        wmap = 1.0 + CFG["node_w_evt"] * evt + CFG["node_w_inter"] * inter
    posm = node_pos_weight(batch, dev)                                                      # E276: tier / visibility of the GT node (None = the E273 loss)
    if posm is not None:
        wmap = posm if wmap is None else wmap * posm
        m = m.float() * posm
    negw = None
    if CFG["neg_w"] is not None:                                                            # E276: a false node costs more near the own lane (zone of the cell)
        z1, z2, z3, z0 = CFG["neg_w"]
        negw = torch.tensor([z0, z1, z2, z3], device=dev)[batch["ztier"].to(dev).long().clamp(0, 3)]
    l_e = focal_bce(no["e"].float(), ex, norm=npos, pos_w=wmap, neg_w=negw)
    fl = batch["flags"].to(dev).float()
    l_f = (F.binary_cross_entropy_with_logits(no["flags"].float(), fl, reduction="none", pos_weight=torch.tensor(6.0, device=dev)) * m[:, None]).sum() / npos
    off_t = batch[off_key].to(dev).float()
    l_o = ((no["off"].float() - off_t).abs().sum(1) * m).sum() / npos
    dr = QN.unit(no["dir_raw"].float())
    l_d = ((1 - (dr * batch[dir_key].to(dev).float()).sum(1)) * m).sum() / npos
    l_s = (F.binary_cross_entropy_with_logits(no["s"].float(), batch["sstart"].to(dev).float(), reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * m).sum() / npos
    l_u = sum((F.binary_cross_entropy_with_logits(no[k].float(), batch[k].to(dev).float(), reduction="none", pos_weight=torch.tensor(8.0, device=dev)) * m).sum() / npos for k in ("uout", "uin"))
    info.update(e=float(l_e), flags=float(l_f), off=float(l_o), dir=float(l_d), s=float(l_s), u=float(l_u))
    if zinfo is not None:
        info.update(zones=float(zinfo["n_zones"]), win_off_gt=float(zinfo["win_off_gt"]))
    return l_e + 0.5 * l_f + l_o + 0.5 * l_d + 0.5 * l_s + 2.0 * l_u, info


def _seg_lse(z, gid, G, extra):
    """log(sum_j exp(z_j) + exp(extra)) per group (a segment log-sum-exp with one fixed 'none' logit per group)"""
    m = torch.full((G,), float("-inf"), device=z.device).scatter_reduce(0, gid, z, "amax", include_self=True)
    m = torch.maximum(m, torch.full_like(m, extra))
    s = torch.zeros(G, device=z.device).scatter_add(0, gid, torch.exp(z - m[gid])) + math.exp(extra) * torch.exp(-m)
    return m + torch.log(s)


def long_match_loss(L, none_logit=0.0):
    """Dual softmax over the long-link candidates (SuperGlue / LoFTR style, one 'none' entry per row and per column, no iterations): the candidates of one source node compete, so do the
    candidates of one destination node. Multi-positive NLL (a node may have several true successors: the mass of the true ones adds up); a node without a true long link goes to 'none'.
    Training signal only: the read-out still thresholds the independent link probability."""
    lg = L["is_long"]
    z = L["logit"].float()[lg]
    y = (L["label"][lg] > 0.5).float()
    K = L["valid"].shape[1]
    total = z.new_zeros(())
    for gid_ in (L["pb"][lg] * K + L["pa"][lg], L["pb"][lg] * K + L["pc"][lg]):                 # rows: the source node, columns: the destination node
        _, inv = torch.unique(gid_, return_inverse=True)
        G = int(inv.max()) + 1
        lse = _seg_lse(z, inv, G, none_logit)
        has = torch.zeros(G, device=z.device).scatter_add(0, inv, y) > 0
        lse_pos = _seg_lse(torch.where(y > 0, z, torch.full_like(z, -1e4)), inv, G, -1e4)       # log of the summed weight of the true candidates only (no clamp: no dead zone)
        nll = torch.where(has, lse - lse_pos, lse - none_logit)                                  # -log(mass of the true ones) = lse - lse_pos; -log(P(none)) = lse - none_logit
        total = total + nll.mean()
    return total / 2


def link_loss(L, batch):
    dev = L["logit"].device
    y = L["label"]
    npos = y.sum().clamp_min(1.0)
    info = {}
    lg = L["is_long"]
    sh = ~lg
    lw = None                                                                                     # E276: the weight of every candidate pair by the tier / visibility of its source node (None = E273)
    if CFG["link_tier_w"] is not None or CFG["vis_w"] != 1.0:
        tier_act = batch["tier"].to(dev).flatten(1).gather(1, L["act"]).long().clamp(0, 3)
        w_act = tier_lut(CFG["link_tier_w"], dev)[tier_act] if CFG["link_tier_w"] is not None else torch.ones(tier_act.shape, device=dev)
        lw = w_act[L["pb"], L["pa"]]
        if CFG["vis_w"] != 1.0:                                                                   # a missing real link out of an unseen node is not demanded (positives x vis_w), a FALSE link out of it still costs (negatives keep the tier weight): the card's "no demand, but a false one is punished"
            va = batch["vis"].to(dev).flatten(1).gather(1, L["act"]).float()
            vf = (va + (1.0 - va) * CFG["vis_w"])[L["pb"], L["pa"]]
            lw = lw * torch.where(y > 0.5, vf, torch.ones_like(vf))
    l_b = focal_bce(L["logit"].float()[sh], y[sh], norm=y[sh].sum().clamp_min(1.0), w=None if lw is None else lw[sh])               # short links: focal BCE over all candidate pairs
    l_long = torch.zeros((), device=dev)
    if lg.any():                                                                                  # long links: their positives are ~5 % of all positives and ~0.3 % of their own candidates: a balanced BCE of their own
        yl = y[lg]
        npl, nnl = yl.sum(), (1 - yl).sum()
        wgt = torch.where(yl > 0.5, (nnl / npl.clamp_min(1.0)).clamp(1.0, 15.0), torch.ones_like(yl))
        if lw is not None:
            wgt = wgt * lw[lg]
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
    l_m = torch.zeros((), device=dev)
    if CFG["long_match"] > 0 and lg.any():                                                       # E274: the long-link candidates of one entry (or one exit) compete, with a 'none' entry
        l_m = long_match_loss(L)
        info["match"] = float(l_m)
    return l_b + 2.0 * l_long + 2.0 * l_sc + CFG["long_match"] * l_m, info                                  # the degree term is logged, not trained: with one-to-many labels a node has several true successors


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
    if phase == 8:                                                                               # E284: the end head
        return LQEN.ends_loss(model.ends_forward(batch, gt=True), batch)
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

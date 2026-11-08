"""The lane head as ONE module (E273): pixel maps -> nodes -> links (Hermite curves) -> sections -> map assignment, and the one loss it is trained with, phase by phase.

    model = QGraph(width=48)
    out = model(batch, upto="assign", frozen=("pyr", "nodes"))          # stages after `upto` are not run; stages in `frozen` run under no_grad
    loss, info = qgraph_loss(model, batch, phase)                       # phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint

Stages: pyr (`pyramid.PyramidHead`), nodes (`qnodes.NodeHead`), links (`qlinks.LinkHead`), sections (`sections.section_membership`, no parameters beyond the node head's s output),
assign (`qassign.MapEncoder` + `qassign.Assigner`). The forward path is made of soft steps; the index choices (active nodes, long-link endpoints) are named in `qlinks`; the thresholds of the
output step (e > 0.5, link > 0.5, argmax of the section membership) are in `evaluation/qgraph_eval.py`.
The losses of every stage are in this file; terms are logged separately (`info`).
Run the self-test: python qgraph.py
"""
import math

import torch
import torch.nn.functional as F

import losses as GL
import qassign as QA
import qlinks as QL
import qnodes as QN
import sections as SE
from pyramid import PyramidHead

STAGES = ("pyr", "nodes", "links", "sections", "assign")
PHASE_UPTO = {2: "pyr", 3: "nodes", 4: "links", 5: "sections", 6: "assign", 7: "assign"}
PHASE_TRAIN = {2: ("pyr",), 3: ("nodes",), 4: ("links",), 5: ("nodes_s",), 6: ("menc", "asg"), 7: ("pyr", "nodes", "links", "menc", "asg")}
PHASE_FROZEN = {2: (), 3: ("pyr",), 4: ("pyr", "nodes"), 5: ("pyr", "links"), 6: ("pyr", "nodes", "links"), 7: ()}
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


class QGraph(torch.nn.Module):
    def __init__(self, width=48, f05=32, d=64, soft_nms=None):
        """soft_nms = None (the E273 head) or (radius m, tau m, passes) of `qnodes.soft_suppress_logit` (E277: the close-node suppression as a soft step right after the node head: the existence
        logit `no['e']` that every later stage, the loss and the read-out see is the damped one; the head's own logit stays in `no['e_raw']`). A property of the model, not of the loss: it travels
        with the checkpoint (`args['soft_nms']`, read by `qpredict.load_model`)."""
        super().__init__()
        self.soft_nms = tuple(float(v) for v in soft_nms) if soft_nms else None
        assert self.soft_nms is None or (len(self.soft_nms) == 3 and 0.0 < self.soft_nms[0] <= 2.0 and self.soft_nms[1] > 0.0 and self.soft_nms[2] >= 1 and self.soft_nms[2] == int(self.soft_nms[2])), soft_nms
        self.pyr = PyramidHead(width=width, f05=f05)
        self.nodes = QN.NodeHead(c2=3 * width, c1=width, c05=f05, d=d)
        self.links = QL.LinkHead(d=d)
        self.menc = QA.MapEncoder(d=d)
        self.asg = QA.Assigner(d=d)

    def node_stage(self, P):
        """the node head and, when `soft_nms` is set, the soft close-node step (E277: `no['e']` is then the damped logit, the head's own one stays in `no['e_raw']`); the ONE place that the forward path,
        the gates (`qgates._node_stats`) and the timing script call, so that nothing reads the raw head by mistake (critic pass 2026-11-08)"""
        no = self.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
        if self.soft_nms:
            no = dict(no)
            no["e_raw"] = no["e"]
            no["e"] = QN.soft_suppress_logit(no["e"], no["pos"], self.soft_nms[0], self.soft_nms[1], int(self.soft_nms[2]))
        return no

    def forward(self, batch, upto="assign", frozen=(), gt=True, rounds=2):
        dev = next(self.parameters()).device
        x = batch["x"].to(dev, non_blocking=True).float()
        out = {}
        stage_i = STAGES.index(upto)

        def run(name, fn, *a, **k):
            if name in frozen:
                with torch.no_grad():
                    return fn(*a, **k)
            return fn(*a, **k)

        P = run("pyr", self.pyr, x)
        out["P"] = P
        if stage_i < 1:
            return out
        no = run("nodes", self.node_stage, P)
        out["no"] = no
        if stage_i < 2:
            return out
        cen = P["centre1"][:, :1].sigmoid()
        gtd = {k: batch[k].to(dev) for k in ("exists", "uout", "uin", "lk", "offset")} if gt else None
        L = run("links", self.links, no, cen, gt=gtd)
        out["L"] = L
        if stage_i < 3:
            return out
        with torch.autocast(device_type="cuda", enabled=False):
            M, Pb = SE.section_membership({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in L.items()}, no["s"].float())
        out["M"], out["Pb"] = M, Pb
        if stage_i < 4:
            return out
        mp = dict(map_pos=batch["map_pos"].to(dev), map_tan=batch["map_tan"].to(dev), map_frac=batch["map_frac"].to(dev), map_pidx=batch["map_pidx"].to(dev), map_pfeat=batch["map_pfeat"].to(dev),
                  map_adj=batch["map_adj"].to(dev), kmask=batch["kmask"].to(dev), tmask=batch["tmask"].to(dev), map_road=batch["map_road"].to(dev))
        with torch.autocast(device_type="cuda", enabled=False):
            enc = self.menc(mp["map_pos"], mp["map_tan"], mp["map_frac"], mp["map_pidx"], mp["map_pfeat"], mp["map_adj"], mp["kmask"], mp["tmask"])
            e_act = no["e"].float().flatten(1).gather(1, L["act"]).sigmoid()
            A = self.asg(L["tok"].float(), L["pos"].float(), L["dirv"].float(), e_act.detach() if "nodes" in frozen else e_act, L["valid"], mp, enc, rounds=rounds)
        out["enc"], out["A"], out["mp"] = enc, A, mp
        return out


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
    out = model(batch, upto=PHASE_UPTO[phase], frozen=PHASE_FROZEN[phase])
    loss, info = 0.0, {}
    if phase in (2, 7):
        l, i = pixel_loss(out["P"], batch)
        loss, info = loss + W["pix"] * l, {**info, **{"px_" + k: v for k, v in i.items()}}
    if phase in (3, 5, 7):
        l, i = node_loss(out["no"], batch)
        loss, info = loss + W["node"] * l, {**info, **{"n_" + k: v for k, v in i.items()}}
    if phase in (4, 7):
        l, i = link_loss(out["L"], batch)
        loss, info = loss + W["link"] * l, {**info, **{"l_" + k: v for k, v in i.items()}}
    if phase in (5, 7):
        ce, pl = SE.section_loss(out["M"], dict(piece=batch["piece"].to(out["L"]["act"].device), sstart=batch["sstart"].to(out["L"]["act"].device)), out["L"])
        loss, info = loss + W["sec"] * (ce + pl), {**info, "s_ce": float(ce), "s_pair": float(pl)}
    if phase in (6, 7):
        l, i = assign_loss(out["A"], out["L"], batch)
        loss, info = loss + W["asg"] * l, {**info, **{"a_" + k: v for k, v in i.items()}}
    return loss, info


def trainable_params(model, phase):
    groups = dict(pyr=model.pyr, nodes=model.nodes, links=model.links, menc=model.menc, asg=model.asg)
    names = PHASE_TRAIN[phase]
    ps = []
    for n in names:
        if n == "nodes_s":
            ps += [p for k, p in model.nodes.named_parameters()]                               # the s head lives in the node head: phase 5 fits the node head (not the pixels / links)
        else:
            ps += list(groups[n].parameters())
    return ps


def _selftest():
    torch.manual_seed(0)
    B = 2
    m = QGraph(width=16, f05=16, d=32)
    # a synthetic batch with the real shapes
    import numpy as np
    import qtargets as QT
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A_ = QT._piece("A", ys(-40, 0), succ=["B"], start_open=True)
    B_ = QT._piece("B", ys(0, 40), pred=["A"], end_open=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["B"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A_, B_], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]), sample="x")
    import map_sim as MS
    nt = QT.node_table(g)
    gt = QT.grid_targets(nt)
    lk, lkl = QT.link_array(nt)
    mp = MS.make_map(g, "exact")
    pm, pr = MS.piece_targets(g, mp), MS.road_targets(g, mp)
    item = dict(x=torch.randn(39, 200, 200).half(), row=0, k=0, token="t", lk=torch.from_numpy(lk), lk_long=torch.from_numpy(lkl), piece_map=pm, piece_road=pr, mp=mp, n_pieces=2,
                g05=torch.from_numpy(np.concatenate([p["pts"] for p in g["pieces"]]).astype(np.float32)), t05=torch.zeros(len(A_["pts"]) + len(B_["pts"]), 2),
                lines_mask=torch.zeros(3, 200, 200), lines_dist=torch.full((3, 200, 200), 2.0), events1=torch.zeros(6, 100, 100), road2=torch.zeros(2, 50, 50))
    item["t05"][:, 1] = 1.0
    item.update({k: torch.from_numpy(v) for k, v in gt.items()})
    item["lk_scale"] = torch.full((len(lk), 2), float("nan"))
    import sys
    sys.path.insert(0, "../data")
    import lane_gt_data as LD
    b = LD.collate([item, item])
    b["lk_scale"] = torch.full((B, b["lk"].shape[1], 2), float("nan"))
    for phase in (2, 3, 4, 5, 6, 7):
        m.zero_grad()
        loss, info = qgraph_loss(m, b, phase)
        assert torch.isfinite(loss), (phase, info)
        loss.backward()
        ps = trainable_params(m, phase)
        got = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        assert got > 0, f"phase {phase}: no gradient reaches the trainable parameters"
        frozen_got = [n for n in PHASE_FROZEN[phase] if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in {"pyr": m.pyr, "nodes": m.nodes, "links": m.links}[n].parameters())]
        assert not frozen_got, f"phase {phase}: frozen stage {frozen_got} got a gradient"
        print("phase", phase, "loss", round(float(loss), 3), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(info.items())[:8]})
    base_losses = {ph: float(qgraph_loss(m, b, ph)[0]) for ph in (2, 3, 4)}                       # the E273 loss (switches at their defaults)
    # E274 switches: finite loss and a gradient with the recall weights and the long-link matching on
    CFG.update(ev_w=3.0, node_w_evt=2.0, node_w_inter=1.0, long_match=0.5)
    try:
        for phase in (2, 3, 4):
            m.zero_grad()
            loss, info = qgraph_loss(m, b, phase)
            assert torch.isfinite(loss), ("E274 switches", phase, info)
            loss.backward()
            assert sum(float(p.grad.abs().sum()) for p in trainable_params(m, phase) if p.grad is not None) > 0, f"E274 switches, phase {phase}: no gradient"
        # the weight of the positives raises the loss of a missed positive and leaves the negatives alone
        lg_, y_ = torch.full((1, 4), -3.0), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        base = focal_bce(lg_, y_, norm=1.0)
        heavy = focal_bce(lg_, y_, norm=1.0, pos_w=torch.tensor([[3.0, 3.0, 3.0, 3.0]]))
        ones = torch.ones(1, 4)
        assert abs(float(focal_bce(lg_, ones, norm=1.0, pos_w=torch.tensor([[3.0, 3.0, 3.0, 3.0]]))) / float(focal_bce(lg_, ones, norm=1.0)) - 3.0) < 1e-5     # the weight scales the positive term exactly
        assert float(heavy) > float(base)
        only_neg = focal_bce(lg_, torch.zeros(1, 4), norm=1.0)
        assert abs(float(focal_bce(lg_, torch.zeros(1, 4), norm=1.0, pos_w=torch.tensor([[1.0, 3.0, 3.0, 3.0]]))) - float(only_neg)) < 1e-6      # and leaves the negatives alone
        # the weight map of node_loss: an event cell and an intersection cell weigh 1 + w_evt / 1 + w_inter, other cells 1 (real channels: flags (B, 8, 50, 50), road2[:, 1] = intersection)
        nb = dict(exists=torch.zeros(1, 50, 50), flags=torch.zeros(1, 8, 50, 50), road2=torch.zeros(1, 2, 50, 50), offset=torch.zeros(1, 2, 50, 50), dirv=torch.zeros(1, 2, 50, 50), sstart=torch.zeros(1, 50, 50),
                  uout=torch.zeros(1, 50, 50), uin=torch.zeros(1, 50, 50))
        nb["exists"][0, 5, 5] = nb["exists"][0, 6, 6] = nb["exists"][0, 7, 7] = 1.0
        nb["flags"][0, 2, 5, 5] = 1.0                                                                 # an event at (5, 5)
        nb["road2"][0, 1, 6, 6] = 1.0                                                                 # an intersection cell at (6, 6)
        nb["dirv"][0, 0] = 1.0
        no_ = dict(e=torch.full((1, 50, 50), -2.0, requires_grad=True), flags=torch.zeros(1, 8, 50, 50), off=torch.zeros(1, 2, 50, 50), dir_raw=torch.ones(1, 2, 50, 50), s=torch.zeros(1, 50, 50),
                   uout=torch.zeros(1, 50, 50), uin=torch.zeros(1, 50, 50))
        CFG.update(node_w_evt=2.0, node_w_inter=1.0)
        node_loss(no_, nb)[0].backward()
        gr = no_["e"].grad[0].abs()
        assert abs(float(gr[5, 5]) / float(gr[7, 7]) - 3.0) < 1e-3 and abs(float(gr[6, 6]) / float(gr[7, 7]) - 2.0) < 1e-3, (float(gr[5, 5]), float(gr[6, 6]), float(gr[7, 7]))
        # dual softmax: small for the right logits, large for the wrong ones, a source without a true link goes to 'none'
        pa = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        pc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        lab = torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0, 0])                                     # source 0 -> 0, source 1 -> 1, source 2 has no true link
        Ld = dict(is_long=torch.ones(9, dtype=torch.bool), label=lab, pb=torch.zeros(9, dtype=torch.long), pa=pa, pc=pc, valid=torch.ones(1, 4, dtype=torch.bool))
        good = long_match_loss(dict(Ld, logit=torch.where(lab > 0, 6.0, -6.0)))
        wrong = long_match_loss(dict(Ld, logit=torch.where(lab > 0, -6.0, 6.0)))
        assert 0.0 <= float(good) < 0.1 and float(wrong) > 3.0, (float(good), float(wrong))
        # a source with two true successors: the mass of the two adds up (loss small when each has about half); two frames are kept apart (a missing batch index in the group key would mix them)
        two = dict(is_long=torch.ones(6, dtype=torch.bool), label=torch.tensor([1.0, 1.0, 0, 1.0, 0, 0]), pb=torch.tensor([0, 0, 0, 1, 1, 1]), pa=torch.zeros(6, dtype=torch.long),
                   pc=torch.tensor([0, 1, 2, 0, 1, 2]), valid=torch.ones(2, 4, dtype=torch.bool))
        lz = torch.tensor([3.0, 3.0, -9.0, 6.0, -9.0, -9.0])
        assert float(long_match_loss(dict(two, logit=lz))) < 0.5
        # a dead zone: a true logit far below its competitors still gets a gradient
        gz = torch.tensor([-14.5, 2.0, 2.0], requires_grad=True)
        long_match_loss(dict(is_long=torch.ones(3, dtype=torch.bool), label=torch.tensor([1.0, 0, 0]), pb=torch.zeros(3, dtype=torch.long), pa=torch.zeros(3, dtype=torch.long),
                             pc=torch.tensor([0, 1, 2]), valid=torch.ones(1, 4, dtype=torch.bool), logit=gz)).backward()
        assert float(gz.grad[0].abs()) > 0.1, float(gz.grad[0])
        g = torch.where(lab > 0, 0.0, 0.0).requires_grad_(True)
        long_match_loss(dict(Ld, logit=g)).backward()
        assert float(g.grad.abs().sum()) > 0
    finally:
        CFG.update(ev_w=1.0, node_w_evt=0.0, node_w_inter=0.0, long_match=0.0)
    for ph in (2, 3, 4):
        again = float(qgraph_loss(m, b, ph)[0])
        assert abs(again - base_losses[ph]) < 1e-6 * max(1.0, abs(again)), (ph, base_losses[ph], again)         # switched off again = the E273 loss
    print("E274 switches ok: weights, dual softmax, defaults give the E273 loss")
    # E277: the soft close-node step: every phase runs with a finite loss and the gradient reaches the trainable groups (phase 3: through the damped logit into the node head)
    ms = QGraph(width=16, f05=16, d=32, soft_nms=(1.5, 0.1, 3))
    ms.load_state_dict(m.state_dict())
    for phase in (3, 4, 5, 6, 7):
        ms.zero_grad()
        loss, info = qgraph_loss(ms, b, phase)
        assert torch.isfinite(loss), ("soft_nms", phase, info)
        loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in trainable_params(ms, phase) if p.grad is not None) > 0, f"soft_nms, phase {phase}: no gradient reaches the trainable parameters"
    print("E277 soft close-node step ok: every phase finite, the gradient reaches the trainable groups")
    print("qgraph selftest ok")


if __name__ == "__main__":
    _selftest()

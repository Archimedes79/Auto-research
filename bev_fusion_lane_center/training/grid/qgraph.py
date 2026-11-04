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


class QGraph(torch.nn.Module):
    def __init__(self, width=48, f05=32, d=64):
        super().__init__()
        self.pyr = PyramidHead(width=width, f05=f05)
        self.nodes = QN.NodeHead(c2=3 * width, c1=width, c05=f05, d=d)
        self.links = QL.LinkHead(d=d)
        self.menc = QA.MapEncoder(d=d)
        self.asg = QA.Assigner(d=d)

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
        no = run("nodes", self.nodes, P["feat2"], P["feat1"], P["feat05"], P["road2"])
        out["no"] = no
        if stage_i < 2:
            return out
        cen = P["centre1"][:, :1].sigmoid()
        gtd = {k: batch[k].to(dev) for k in ("exists", "uout", "uin", "lk")} if gt else None
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
    l_f = (F.binary_cross_entropy_with_logits(no["flags"].float(), fl, reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * m[:, None]).sum() / (npos * 8)
    off_t = batch["offset"].to(dev).float()
    l_o = ((no["off"].float() - off_t).abs().sum(1) * m).sum() / npos
    dr = QN.unit(no["dir_raw"].float())
    l_d = ((1 - (dr * batch["dirv"].to(dev).float()).sum(1)) * m).sum() / npos
    l_s = (F.binary_cross_entropy_with_logits(no["s"].float(), batch["sstart"].to(dev).float(), reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * m).sum() / npos
    l_u = sum((F.binary_cross_entropy_with_logits(no[k].float(), batch[k].to(dev).float(), reduction="none", pos_weight=torch.tensor(8.0, device=dev)) * m).sum() / npos for k in ("uout", "uin"))
    info.update(e=float(l_e), flags=float(l_f), off=float(l_o), dir=float(l_d), s=float(l_s), u=float(l_u))
    return l_e + l_f + l_o + 0.5 * l_d + 0.5 * l_s + 0.5 * l_u, info


def link_loss(L, batch):
    dev = L["logit"].device
    y = L["label"]
    npos = y.sum().clamp_min(1.0)
    info = {}
    l_b = focal_bce(L["logit"].float(), y, norm=npos)
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
    info.update(link=float(l_b), deg=float(l_deg), scale=float(l_sc), pos=int(npos), pairs=len(y), miss=int(L.get("gt_missing", 0)))
    return l_b + 0.5 * l_deg + 2.0 * l_sc, info


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
    print("qgraph selftest ok")


if __name__ == "__main__":
    _selftest()

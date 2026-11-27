"""The lane head as ONE module (E273): pixel maps -> nodes -> links (Hermite curves) -> sections -> map assignment, and the one loss it is trained with, phase by phase.

    model = QGraph(width=48)
    out = model(batch, upto="assign", frozen=("pyr", "nodes"))          # stages after `upto` are not run; stages in `frozen` run under no_grad
    loss, info = loss_qgraph.qgraph_loss(model, batch, phase)           # phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint | 8 end head | 9 joint without the pixel stage (E292)

Stages: pyr (`pyramid.PyramidHead`), nodes (`qnodes.NodeHead`), links (`qlinks.LinkHead`), sections (`sections.section_membership`, no parameters beyond the node head's s output),
assign (`qassign.MapEncoder` + `qassign.Assigner`). The forward path is made of soft steps; the index choices (active nodes, long-link endpoints) are named in `qlinks`; the thresholds of the
output step (e > 0.5, link > 0.5, argmax of the section membership) are in `evaluation/qgraph_eval.py`.
The losses of every stage are in net/loss/loss_qgraph.py; terms are logged separately (`info`).
Run the self-test (from the problem folder): python net/model/qgraph.py
"""

import torch

import qassign as QA
import qendnet as QEN
import qlinks as QL
import qmnms as QMN
import qnodes as QN
import sections as SE
from pyramid import PyramidHead

STAGES = ("pyr", "nodes", "links", "sections", "assign")
PHASE_TRAIN = {2: ("pyr",), 3: ("nodes",), 4: ("links",), 5: ("nodes_s",), 6: ("menc", "asg"), 7: ("pyr", "nodes", "links", "menc", "asg")}
PHASE_FROZEN = {2: (), 3: ("pyr",), 4: ("pyr", "nodes"), 5: ("pyr", "links"), 6: ("pyr", "nodes", "links"), 7: ()}
# E284: phase 8 trains the junction END head (`qendnet`) on the frozen pixel and node stages; the model must be built with ends=True. It is a side stage: B's checkpoint, forward path and numbers are untouched.
PHASE_TRAIN[8] = ("ends",)
PHASE_FROZEN[8] = ("pyr", "nodes")
# E292: phase 9 is the joint phase 7 without the pixel stage: nodes, links, map encoder and assigner train together on a FROZEN pyramid (the fine-tune of B's final checkpoint, so E312's maps and pair head stay valid)
PHASE_TRAIN[9] = ("nodes", "links", "menc", "asg")
PHASE_FROZEN[9] = ("pyr",)
class QGraph(torch.nn.Module):
    def __init__(self, width=48, f05=32, d=64, soft_nms=None, ends=False):
        """soft_nms = None (the E273 head) or (radius m, tau m, passes) of `qnodes.soft_suppress_logit` (E277: the close-node suppression as a soft step right after the node head: the existence
        logit `no['e']` that every later stage, the loss and the read-out see is the damped one; the head's own logit stays in `no['e_raw']`) or ('mnms', radius m, sigma) of `qmnms.matrix_nms_logit`
        (E292: the one-pass Matrix-NMS decay in the same place, with a hard neighbour ORDER and a smooth decision to drop; no parameters). A property of the model, not of the loss: it travels
        with the checkpoint (`args['soft_nms']`, read by `qpredict.load_model`; the Matrix-NMS form is a list that starts with the string 'mnms')."""
        super().__init__()
        self.mnms = None
        if soft_nms and soft_nms[0] == "mnms":
            assert len(soft_nms) == 3, soft_nms
            self.mnms = (float(soft_nms[1]), float(soft_nms[2]))                                  # (radius m, sigma): the asserts of `qmnms.matrix_nms_logit` apply when it runs, here at build time
            assert 0.0 < self.mnms[0] <= 2.0 and self.mnms[1] > 0.0, soft_nms
            soft_nms = None
        self.soft_nms = tuple(float(v) for v in soft_nms) if soft_nms else None
        assert self.soft_nms is None or (len(self.soft_nms) == 3 and 0.0 < self.soft_nms[0] <= 2.0 and self.soft_nms[1] > 0.0 and self.soft_nms[2] >= 1 and self.soft_nms[2] == int(self.soft_nms[2])), soft_nms
        self.pyr = PyramidHead(width=width, f05=f05)
        self.nodes = QN.NodeHead(c2=3 * width, c1=width, c05=f05, d=d)
        self.links = QL.LinkHead(d=d)
        self.menc = QA.MapEncoder(d=d)
        self.asg = QA.Assigner(d=d)
        self.ends = QEN.EndHead(d=d) if ends else None                                          # E284: the junction end head (phase 8); None = the E273 head, whose checkpoints have no `ends.*` tensors

    def ends_forward(self, batch, gt=False):
        """E284: the pixel and node stages under no_grad, then the end head on the node tokens; gt = the batch's GT ends are added to the keypoint candidates (training)"""
        assert self.ends is not None, "the model was built without the end head (QGraph(ends=True))"
        dev = next(self.parameters()).device
        x = batch["x"].to(dev, non_blocking=True).float()
        with torch.no_grad():
            no = self.node_stage(self.pyr(x))
        g = {k: batch[k].to(dev) for k in ("ent", "ext", "ent_m", "ext_m")} if gt else None
        return self.ends(no, gt=g)

    def node_stage(self, P):
        """the node head and, when `soft_nms` is set, the soft close-node step (E277: `no['e']` is then the damped logit, the head's own one stays in `no['e_raw']`); the ONE place that the forward path,
        the gates (`qgates._node_stats`) and the timing script call, so that nothing reads the raw head by mistake (critic pass 2026-11-08)"""
        no = self.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
        if self.soft_nms:
            no = dict(no)
            no["e_raw"] = no["e"]
            no["e"] = QN.soft_suppress_logit(no["e"], no["pos"], self.soft_nms[0], self.soft_nms[1], int(self.soft_nms[2]))
        elif self.mnms:                                                                           # E292: the same place, the Matrix-NMS decay (e_raw = the head's own logit)
            no = dict(no)
            no["e_raw"] = no["e"]
            no["e"] = QMN.matrix_nms_logit(no["e"], no["pos"], self.mnms[0], self.mnms[1])
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


def trainable_params(model, phase):
    groups = dict(pyr=model.pyr, nodes=model.nodes, links=model.links, menc=model.menc, asg=model.asg)
    if model.ends is not None:
        groups["ends"] = model.ends
    names = PHASE_TRAIN[phase]
    ps = []
    for n in names:
        if n == "nodes_s":
            ps += [p for k, p in model.nodes.named_parameters()]                               # the s head lives in the node head: phase 5 fits the node head (not the pixels / links)
        else:
            ps += list(groups[n].parameters())
    return ps


def _toy_batch(B=2):
    """a synthetic batch with the real shapes (two lane pieces A -> B along x = 0, two window events); the batch of the selftest and of `test_mnms_layer.py`"""
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
    sys.path.insert(0, "../prep")
    import lane_gt_data as LD
    b = LD.collate([item] * B)
    b["lk_scale"] = torch.full((B, b["lk"].shape[1], 2), float("nan"))
    return b


def _selftest():
    from loss_qgraph import CFG, focal_bce, long_match_loss, node_loss, qgraph_loss
    torch.manual_seed(0)
    B = 2
    m = QGraph(width=16, f05=16, d=32)
    b = _toy_batch(B)
    for phase in (2, 3, 4, 5, 6, 7, 9):
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
    # E292: the Matrix-NMS form: the same switch, no new parameters, every phase finite, the gradient reaches the trainable groups, none the frozen ones (phase 9 included)
    mm = QGraph(width=16, f05=16, d=32, soft_nms=["mnms", 1.75, 0.1])
    assert mm.mnms == (1.75, 0.1) and mm.soft_nms is None and set(mm.state_dict()) == set(m.state_dict())
    mm.load_state_dict(m.state_dict())
    for phase in (3, 4, 5, 6, 7, 9):
        mm.zero_grad()
        loss, info = qgraph_loss(mm, b, phase)
        assert torch.isfinite(loss), ("mnms", phase, info)
        loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in trainable_params(mm, phase) if p.grad is not None) > 0, f"mnms, phase {phase}: no gradient reaches the trainable parameters"
        frozen_got = [n for n in PHASE_FROZEN[phase] if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in {"pyr": mm.pyr, "nodes": mm.nodes, "links": mm.links}[n].parameters())]
        assert not frozen_got, f"mnms, phase {phase}: frozen stage {frozen_got} got a gradient"
    assert {n for n in PHASE_TRAIN[9]} == {"nodes", "links", "menc", "asg"} and PHASE_FROZEN[9] == ("pyr",)
    print("E292 Matrix-NMS layer ok: every phase finite incl. phase 9, the gradient reaches the trainable groups, none the pyramid")
    print("qgraph selftest ok")


if __name__ == "__main__":
    _selftest()

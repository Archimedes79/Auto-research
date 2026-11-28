"""The lane head as ONE module (E273): pixel maps -> nodes -> links (Hermite curves) -> sections -> map assignment, and the one loss it is trained with, phase by phase.

    model = QGraph(width=48)
    out = model(batch, upto="assign", frozen=("pyr", "nodes"))          # stages after `upto` are not run; stages in `frozen` run under no_grad
    loss, info = loss_qgraph.qgraph_loss(model, batch, phase)           # phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint | 9 joint without the pixel stage (E292)

Stages: pyr (`pyramid.PyramidHead`), nodes (`qnodes.NodeHead`), links (`qlinks.LinkHead`), sections (`sections.section_membership`, no parameters beyond the node head's s output),
assign (`qassign.MapEncoder` + `qassign.Assigner`). The forward path is made of soft steps; the index choices (active nodes, long-link endpoints) are named in `qlinks`; the thresholds of the
output step (e > 0.5, link > 0.5, argmax of the section membership) are in `evaluation/qgraph_eval.py`.
The losses of every stage are in net/loss/loss_qgraph.py; terms are logged separately (`info`).
Run the self-test (from the problem folder): python net/model/qgraph.py
"""

import torch

import qassign as QA
import qlinks as QL
import qmnms as QMN
import qnodes as QN
import sections as SE
from pyramid import PyramidHead

STAGES = ("pyr", "nodes", "links", "sections", "assign")
PHASE_TRAIN = {2: ("pyr",), 3: ("nodes",), 4: ("links",), 5: ("nodes_s",), 6: ("menc", "asg"), 7: ("pyr", "nodes", "links", "menc", "asg")}
PHASE_FROZEN = {2: (), 3: ("pyr",), 4: ("pyr", "nodes"), 5: ("pyr", "links"), 6: ("pyr", "nodes", "links"), 7: ()}
# E292: phase 9 is the joint phase 7 without the pixel stage: nodes, links, map encoder and assigner train together on a FROZEN pyramid (the fine-tune of B's final checkpoint, so E312's maps and pair head stay valid)
PHASE_TRAIN[9] = ("nodes", "links", "menc", "asg")
PHASE_FROZEN[9] = ("pyr",)
# Options of the lane head that no longer exist (the tag v1 keeps their code): the value the args of a stored run must hold (or nothing). The loss
# switches of E274 (ev_w, node_w_evt, node_w_inter, long_match), E276 (tier_w, vis_w, neg_w, link_tier_w, the sparse node target), E282 (node_loss
# zone, zone_bias) and the end head of E284; none of the runs that made B' (E271 a, E273 p, E292 a) used one.
REMOVED = dict(ev_w=1.0, node_w_evt=0.0, node_w_inter=0.0, long_match=0.0, tier_w=None, vis_w=1.0, neg_w=None, link_tier_w=None, sparse=None, node_loss="e273", zone_bias=0.05, ends=False)


class QGraph(torch.nn.Module):
    def __init__(self, width=48, f05=32, d=64, soft_nms=None):
        """soft_nms = None (the E273 head) or ('mnms', radius m, sigma) of `qmnms.matrix_nms_logit` (E292: the one-pass Matrix-NMS decay right after the node head, with a hard neighbour ORDER and a smooth decision
        to drop; no parameters: the existence logit `no['e']` that every later stage, the loss and the read-out see is the damped one; the head's own logit stays in `no['e_raw']`). A property of the model, not of
        the loss: it travels with the checkpoint (`args['soft_nms']`, read by `from_args`; a list that starts with the string 'mnms'). The other form of this option, (radius, tau, passes) of E277's soft
        close-node step, is refused: the tag v1 keeps it."""
        super().__init__()
        self.mnms = None
        if soft_nms:
            assert soft_nms[0] == "mnms" and len(soft_nms) == 3, f"soft_nms {soft_nms}: only ('mnms', radius, sigma); the soft close-node step of E277 (radius, tau, passes) no longer exists (the tag v1 keeps it)"
            self.mnms = (float(soft_nms[1]), float(soft_nms[2]))                                  # (radius m, sigma): the asserts of `qmnms.matrix_nms_logit` apply when it runs, here at build time
            assert 0.0 < self.mnms[0] <= 2.0 and self.mnms[1] > 0.0, soft_nms
        self.pyr = PyramidHead(width=width, f05=f05)
        self.nodes = QN.NodeHead(c2=3 * width, c1=width, c05=f05, d=d)
        self.links = QL.LinkHead(d=d)
        self.menc = QA.MapEncoder(d=d)
        self.asg = QA.Assigner(d=d)

    def node_stage(self, P):
        """the node head and, when the Matrix-NMS layer is set, its decay (E292: `no['e']` is then the damped logit, the head's own one stays in `no['e_raw']`); the ONE place that the forward path,
        the gates (`qgates._node_stats`) and the timing script call, so that nothing reads the raw head by mistake (critic pass 2026-11-08)"""
        no = self.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
        if self.mnms:
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


def from_args(args):
    """the lane head of a stored run's args (`width`, `soft_nms`): the ONE place that builds it for a checkpoint (`qpredict.load_model`, `jlive_train.load_b`). Args that hold an option of REMOVED at
    another value are refused: that head or loss no longer exists"""
    held = {k: args[k] for k, v in REMOVED.items() if args.get(k) not in (None, v)}
    assert not held, f"the run's args hold {held}: options of the lane head that no longer exist (the tag v1 keeps them)"
    return QGraph(width=int(args.get("width", 48)), soft_nms=args.get("soft_nms"))


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
    from loss_qgraph import qgraph_loss
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
    # E292: the Matrix-NMS layer: no new parameters, every phase finite, the gradient reaches the trainable groups, none the frozen ones (phase 9 included)
    mm = QGraph(width=16, f05=16, d=32, soft_nms=["mnms", 1.75, 0.1])
    assert mm.mnms == (1.75, 0.1) and set(mm.state_dict()) == set(m.state_dict())
    stored = dict(REMOVED, width=16, soft_nms=["mnms", 1.75, 0.1], phases=[3, 4, 5, 6, 9])            # a stored run's args: removed options at their off values load, another value is refused by name
    assert from_args(stored).mnms == (1.75, 0.1) and from_args({}).mnms is None
    for bad in (dict(ends=True), dict(node_loss="zone"), dict(tier_w=[1.0, 0.5, 0.2]), dict(long_match=0.5), dict(sparse=[0.3, 7.5])):
        try:
            from_args(dict(stored, **bad))
        except AssertionError as e:
            assert list(bad)[0] in str(e) and "tag v1" in str(e), e
        else:
            raise RuntimeError(f"accepted the removed option {bad}")
    try:
        QGraph(width=16, f05=16, d=32, soft_nms=(1.5, 0.1, 3))
    except AssertionError as e:
        assert "E277" in str(e) and "tag v1" in str(e), e
    else:
        raise RuntimeError("accepted the soft close-node step of E277")
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

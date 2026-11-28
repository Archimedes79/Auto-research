"""The lane head as ONE module (E273): pixel maps -> nodes -> links (Hermite curves) -> sections -> map assignment, and the one loss it is trained with, phase
by phase.

    model = QGraph(width=48)
    out = model(batch, upto="assign", frozen=("pyr", "nodes"))          # stages after `upto` are not run; stages in `frozen` run under no_grad
    loss, info = loss_qgraph.qgraph_loss(model, batch, phase)           # phase 2 pixels | 3 nodes | 4 links | 5 sections | 6 assign | 7 joint | 9 joint without
                                                                        # the pixel stage (E292)

Stages: pyr (`pyramid.PyramidHead`), nodes (`qnodes.NodeHead`), links (`qlinks.LinkHead`), sections (`sections.section_membership`, no parameters beyond the
node head's s output), assign (`qassign.MapEncoder` + `qassign.Assigner`). The forward path is made of soft steps; the index choices (active nodes, long-link
endpoints) are named in `qlinks`; the thresholds of the output step (e > 0.5, link > 0.5, argmax of the section membership) are in `evaluation/qgraph_eval.py`.
The losses of every stage are in net/loss/loss_qgraph.py; terms are logged separately (`info`). Test: net/tests/test_qgraph.py
"""

import torch

from lane_center.net.model import qassign as QA
from lane_center.net.model import qlinks as QL
from lane_center.net.model import qmnms as QMN
from lane_center.net.model import qnodes as QN
from lane_center.net.model import sections as SE
from lane_center.net.model.pyramid import PyramidHead

STAGES = ("pyr", "nodes", "links", "sections", "assign")
PHASE_TRAIN = {2: ("pyr",), 3: ("nodes",), 4: ("links",), 5: ("nodes_s",), 6: ("menc", "asg"), 7: ("pyr", "nodes", "links", "menc", "asg")}
PHASE_FROZEN = {2: (), 3: ("pyr",), 4: ("pyr", "nodes"), 5: ("pyr", "links"), 6: ("pyr", "nodes", "links"), 7: ()}
# E292: phase 9 is the joint phase 7 without the pixel stage: nodes, links, map encoder and assigner train together on a FROZEN pyramid (the fine-tune of B's
# final checkpoint, so E312's maps and pair head stay valid)
PHASE_TRAIN[9] = ("nodes", "links", "menc", "asg")
PHASE_FROZEN[9] = ("pyr",)
# Options of the lane head that no longer exist (the tag v1 keeps their code): the value the args of a stored run must hold (or nothing). The loss
# switches of E274 (ev_w, node_w_evt, node_w_inter, long_match), E276 (tier_w, vis_w, neg_w, link_tier_w, the sparse node target), E282 (node_loss
# zone, zone_bias) and the end head of E284; none of the runs that made B' (E271 a, E273 p, E292 a) used one.
REMOVED = dict(ev_w=1.0, node_w_evt=0.0, node_w_inter=0.0, long_match=0.0, tier_w=None, vis_w=1.0, neg_w=None, link_tier_w=None, sparse=None, node_loss="e273",
               zone_bias=0.05, ends=False)


class QGraph(torch.nn.Module):
    def __init__(self, width=48, f05=32, d=64, soft_nms=None):
        """soft_nms = None (the E273 head) or ('mnms', radius m, sigma) of `qmnms.matrix_nms_logit` (E292: the one-pass Matrix-NMS decay right after the node
        head, with a hard neighbour ORDER and a smooth decision to drop; no parameters: the existence logit `no['e']` that every later stage, the loss and the
        read-out see is the damped one; the head's own logit stays in `no['e_raw']`). A property of the model, not of the loss: it travels with the checkpoint
        (`args['soft_nms']`, read by `from_args`; a list that starts with the string 'mnms'). The other form of this option, (radius, tau, passes) of E277's
        soft close-node step, is refused: the tag v1 keeps it."""
        super().__init__()
        self.mnms = None
        if soft_nms:
            assert soft_nms[0] == "mnms" and len(soft_nms) == 3, f"soft_nms {soft_nms}: only ('mnms', radius, sigma); the soft close-node step of E277 (radius, tau, passes) no longer exists (the tag v1 keeps it)"
            # (radius m, sigma): the asserts of `qmnms.matrix_nms_logit` apply when it runs, here at build time
            self.mnms = (float(soft_nms[1]), float(soft_nms[2]))
            assert 0.0 < self.mnms[0] <= 2.0 and self.mnms[1] > 0.0, soft_nms
        self.pyr = PyramidHead(width=width, f05=f05)
        self.nodes = QN.NodeHead(c2=3 * width, c1=width, c05=f05, d=d)
        self.links = QL.LinkHead(d=d)
        self.menc = QA.MapEncoder(d=d)
        self.asg = QA.Assigner(d=d)

    def node_stage(self, P):
        """the node head and, when the Matrix-NMS layer is set, its decay (E292: `no['e']` is then the damped logit, the head's own one stays in `no['e_raw']`);
        the ONE place that the forward path, the gates (`qgates._node_stats`) and the timing script call, so that nothing reads the raw head by mistake (critic
        pass 2026-11-08)"""
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
            A = self.asg(L["tok"].float(), L["pos"].float(), L["dirv"].float(), e_act.detach() if "nodes" in frozen else e_act, L["valid"], mp, enc,
                         rounds=rounds)
        out["enc"], out["A"], out["mp"] = enc, A, mp
        return out


def from_args(args):
    """the lane head of a stored run's args (`width`, `soft_nms`): the ONE place that builds it for a checkpoint (`lane_net.load_b`,
    `lane_net.LaneNet.from_checkpoint`). Args that hold an option of REMOVED at another value are refused: that head or loss no longer exists"""
    held = {k: args[k] for k, v in REMOVED.items() if args.get(k) not in (None, v)}
    assert not held, f"the run's args hold {held}: options of the lane head that no longer exist (the tag v1 keeps them)"
    return QGraph(width=int(args.get("width", 48)), soft_nms=args.get("soft_nms"))


def trainable_params(model, phase):
    groups = dict(pyr=model.pyr, nodes=model.nodes, links=model.links, menc=model.menc, asg=model.asg)
    names = PHASE_TRAIN[phase]
    ps = []
    for n in names:
        if n == "nodes_s":
            # the s head lives in the node head: phase 5 fits the node head (not the pixels / links)
            ps += [p for k, p in model.nodes.named_parameters()]
        else:
            ps += list(groups[n].parameters())
    return ps

"""The whole system: cached BEV features -> soft outputs (one forward, differentiable up to the read-out) -> delivered lane graphs.

    net = LaneGraphNet.from_runs(seed=0)        # in eval mode
    out = net(x)                   # x (B, 39, 200, 200): {"lanes": B's nodes and links, "pairs": keypoints, pair logits, scales[, attached ends]}
    graphs = net.graphs(x)         # forward + the named hard output steps (lg_readout.deliver) -> [Graph]

    x -> Pyramid -> P -> NodeHead -> LinkHead (+ section membership) ................. B's lane pieces
                    P -> JunctionMaps -> keypoints -> PairHead (order rule, evidence) -> connectors
                                         NodeHead outputs -> Attach (only for Config.attached_ends)
"""
import numpy as np
import torch
import torch.nn as nn

from lg_attach import Attach
from lg_config import CHAMPION, Config, run_paths
from lg_graph import Graph
from lg_junction_maps import JunctionMaps, keypoints
from lg_lane_head import LinkHead, NodeHead
from lg_pair_head import PairHead
from lg_pyramid import Pyramid
from lg_readout import LaneFrame, PairFrame, deliver

# the one place where checkpoint names meet module names: (checkpoint prefix, module prefix or None = not loaded), first match wins.
# Not loaded: the pyramid's 0.5 m line maps and B's map assignment (not part of the delivered graph), and the junction run's copies of
# the frozen pyramid and node head (checked equal to B's).
B_KEYS = (("pyr.lines05.", None), ("pyr.", "pyramid."), ("nodes.", "nodes."), ("links.", "links."), ("menc.", None), ("asg.", None))
J_KEYS = (("jmodel.pyr.", None), ("nodes.", None), ("jmodel.jm.", "maps."), ("je.attach.", "attach."), ("je.", "pairs."))
# the training options the architecture above implements (checked against the checkpoints' args)
B_ARGS = dict(width=48, soft_nms=None, ends=None)
J_ARGS = dict(no_order=False, no_evidence=False, scales=True, ntok=False, unc=False, attach=True)
MAPS_ARGS = dict(ctx="dil", d=64)


def rename(state: dict, rules: tuple) -> dict:
    out = {}
    for k, v in state.items():
        rule = next((r for r in rules if k.startswith(r[0])), None)
        if rule is None:
            raise KeyError(f"unknown checkpoint tensor {k}")
        if rule[1]:
            out[rule[1] + k[len(rule[0]):]] = v
    return out


def check_args(args: dict, want: dict, what: str) -> None:
    bad = {k: (args.get(k), v) for k, v in want.items() if args.get(k) != v}
    if bad:
        raise ValueError(f"{what} was trained with options this code does not implement (found, expected): {bad}")


class LaneGraphNet(nn.Module):
    def __init__(self, cfg: Config = CHAMPION):
        super().__init__()
        self.cfg = cfg
        self.pyramid, self.nodes, self.links = Pyramid(), NodeHead(), LinkHead(cfg)
        self.maps, self.pairs, self.attach = JunctionMaps(), PairHead(cfg), Attach(cfg)

    def forward(self, x: torch.Tensor) -> dict:
        P = self.pyramid(x)
        no = self.nodes(P)
        lanes = self.links(no, P)
        jm = self.maps(P)
        kp = keypoints(jm, self.cfg)
        pairs = self.pairs(kp, P, jm)
        if self.cfg.attached_ends:
            att = self.attach(pairs["tok"], kp, no)
            pairs.update(att_pos=att["pos"], att_dir=att["dir"], att_p=att["p"])
        pairs.update(kp_pos=kp["pos"], kp_dir=kp["dir"], kp_p=kp["p"], valid=kp["valid"])
        return dict(lanes=lanes, pairs=pairs)

    @torch.no_grad()
    def graphs(self, x: torch.Tensor) -> list[Graph]:
        return [deliver(lanes, pairs, self.cfg) for lanes, pairs in per_frame(self(x))]

    @classmethod
    def from_runs(cls, seed: int, cfg: Config = CHAMPION) -> "LaneGraphNet":
        """the champion of one seed from its two training runs (`runs/e273_p_s<seed>`, `runs/e324_s_s<seed>`), in eval mode; raises when a run was
        trained with other options or the junction run does not sit on B's frozen pyramid and node head"""
        b, j = (torch.load(p, map_location="cpu") for p in run_paths(seed))
        check_args(b["args"], B_ARGS, f"B (seed {seed})")
        check_args(j["args"], J_ARGS, f"the junction run (seed {seed})")
        check_args(j["args"]["maps_args"], MAPS_ARGS, f"the junction maps (seed {seed})")
        b, j = b["state"], j["state"]
        for k in (k for k in j if k.startswith(("jmodel.pyr.", "nodes."))):
            if not torch.equal(j[k], b[k.replace("jmodel.pyr.", "pyr.")]):
                raise ValueError(f"{k}: the junction run does not sit on B's frozen stage")
        net = cls(cfg)
        net.load_state_dict({**rename(b, B_KEYS), **rename(j, J_KEYS)})
        return net.eval()


def per_frame(out: dict) -> list[tuple[LaneFrame, PairFrame]]:
    """the batched soft outputs -> one (LaneFrame, PairFrame) per frame (one host copy per tensor)"""
    L = {k: v.detach().cpu() for k, v in out["lanes"].items()}
    p = out["lanes"]["logit"].detach().sigmoid().cpu()                                     # on the device, as the original prediction contract
    Q = {k: v.detach().cpu() for k, v in out["pairs"].items()}
    f64 = lambda t: t.float().numpy().astype(np.float64)  # noqa: E731
    res = []
    for b in range(L["valid"].shape[0]):
        s = L["frame"] == b
        lanes = LaneFrame(pos=f64(L["pos"][b]), dir=f64(L["dir"][b]), e=f64(L["e"][b]), valid=L["valid"][b].numpy(), start=L["start"][b].numpy(), src=L["src"][s].numpy(),
                          dst=L["dst"][s].numpy(), p=f64(p[s]), sa=f64(L["sa"][s]), sb=f64(L["sb"][s]), long=L["long"][s].numpy())
        pairs = PairFrame(**{k: Q[k][b].float().numpy() for k in ("kp_pos", "kp_dir", "logit", "scale", "att_pos", "att_dir") if k in Q}, allowed=Q["allowed"][b].numpy())
        res.append((lanes, pairs))
    return res

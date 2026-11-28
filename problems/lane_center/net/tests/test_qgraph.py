"""Test of the lane head as one module (`net/model/qgraph.QGraph`) with its loss (`net/loss/loss_qgraph.qgraph_loss`) on a toy batch of the real shapes: every phase gives a finite loss, the gradient reaches
the trainable groups of the phase and none of its frozen stages, with and without the Matrix-NMS layer; a stored run's args with a removed option at its off value build the head, another value is refused
(`from_args`). CPU, about a minute."""
import numpy as np
import torch

import lane_gt_data as LD
import map_sim as MS
import qtargets as QT
from loss_qgraph import qgraph_loss
from qgraph import PHASE_FROZEN, PHASE_TRAIN, REMOVED, QGraph, from_args, trainable_params


def toy_batch(B=2):
    """a synthetic batch with the real shapes (two lane pieces A -> B along x = 0, two window events)"""
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A_ = QT._piece("A", ys(-40, 0), succ=["B"], start_open=True)
    B_ = QT._piece("B", ys(0, 40), pred=["A"], end_open=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["B"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A_, B_], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]), sample="x")
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
    b = LD.collate([item] * B)
    b["lk_scale"] = torch.full((B, b["lk"].shape[1], 2), float("nan"))
    return b


def main():
    torch.manual_seed(0)
    B = 2
    m = QGraph(width=16, f05=16, d=32)
    b = toy_batch(B)
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
    print("test_qgraph ok")


if __name__ == "__main__":
    main()

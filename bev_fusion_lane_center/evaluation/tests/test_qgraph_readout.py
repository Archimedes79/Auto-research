"""Test of `qgraph_readout` (the close-node suppression of E275): the rule on hand-made nodes (the cases of the card), an independent brute-force reference on random frames, the properties
(subset, spacing, order of the input, threshold), `apply` (the original prediction is not touched, e of the dropped nodes is 0), and the scorer on the toy lane graph with a duplicate node:
node precision falls with it and is back at 1.0 with the step, a link to the duplicate is not read out any more, the oracle numbers are unchanged.

    wsl bash tools/env/_py14.sh test_qgraph_readout.py
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../../training/grid"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as RO  # noqa: E402


def brute(pos, e, R, thr):
    """independent O(n^2) reference: repeatedly take the strongest node that is still a candidate, keep it, remove the candidates that lie closer than R to it"""
    pos, e = np.asarray(pos, float), np.asarray(e, float)
    cand = [i for i in range(len(e)) if e[i] >= thr]
    keep = np.zeros(len(e), bool)
    while cand:
        best = min(cand, key=lambda i: (-e[i], i))
        keep[best] = True
        cand = [i for i in cand if i != best and np.hypot(*(pos[i] - pos[best])) >= R]
    return keep


def cases():
    k = lambda pos, e, R=1.5, thr=0.5: RO.suppress_close_nodes(np.array(pos, float), np.array(e, float), R, thr).tolist()  # noqa: E731
    assert k([[0, 0], [1.0, 0]], [0.9, 0.8]) == [True, False]                       # two nodes 1.0 m apart: the stronger stays
    assert k([[0, 0], [1.0, 0]], [0.8, 0.9]) == [False, True]
    assert k([[0, 0], [2.0, 0]], [0.9, 0.8]) == [True, True]                        # 2.0 m apart: both stay
    assert k([[0, 0], [1.5, 0]], [0.9, 0.8]) == [True, True]                        # exactly the radius: both stay (strictly closer is suppressed)
    assert k([[0, 0], [1.4999, 0]], [0.9, 0.8]) == [True, False]
    assert k([[0, 0], [1, 0], [2, 0]], [0.9, 0.8, 0.7]) == [True, False, True]      # the chain of the card: the third is 2.0 m from the kept first
    assert k([[0, 0], [1, 0], [2, 0]], [0.8, 0.9, 0.7]) == [False, True, False]     # the middle one is the strongest and suppresses both
    assert k([[0, 0], [1, 0]], [0.8, 0.8]) == [True, False]                         # equal e: the lower index stays
    assert k([[0, 0], [0.5, 0.5]], [0.6, 0.6]) == [True, False]
    assert k([[0, 0], [3.0, 0], [1.5, 0]], [0.9, 0.9, 0.3]) == [True, True, False]  # a node below the threshold is never kept ...
    assert k([[0, 0], [3.0, 0], [1.5, 0]], [0.9, 0.9, 0.3], thr=0.2) == [True, True, True]    # ... with a lower threshold it is a candidate and stays: exactly R from both is not closer than R
    assert k([[0, 0], [3.0, 0], [1.4, 0]], [0.9, 0.9, 0.4], thr=0.3) == [True, True, False]   # closer than R to the first: suppressed
    assert k([[0, 0], [1.0, 0]], [0.4, 0.3]) == [False, False]                      # nothing at or above the threshold
    assert k([[0, 0], [1.0, 0]], [0.9, 0.8], R=0.5) == [True, True]                 # a smaller radius
    assert k([[0, 0], [1.0, 0]], [0.9, 0.8], R=1.0) == [True, True]                 # distance equals the radius
    assert k(np.zeros((0, 2)), np.zeros(0)) == []                                   # empty frame
    assert k([[5, 5]], [0.9]) == [True] and k([[5, 5]], [0.1]) == [False]           # one node
    assert k([[5, 5]], [0.5]) == [True]                                             # e equal to the threshold is delivered (the scorer's kept() reads e >= thr)
    assert k([[0, 0], [1.0, 0]], [0.5, 0.5]) == [True, False]                       # ... and takes part in the suppression
    try:                                                                            # a non-finite position would stop every later suppression of the frame: refused
        k([[0, 0], [np.nan, np.nan], [5, 5], [5.5, 5]], [0.9, 0.8, 0.7, 0.6])
        raise SystemExit("a NaN position was accepted")
    except AssertionError:
        pass
    assert k([[0, 0], [np.nan, np.nan], [5, 5], [5.5, 5]], [0.9, 0.3, 0.7, 0.6]) == [True, False, True, False]   # a NaN node below the threshold is no candidate: fine
    # a weak node below the threshold between two strong nodes more than R apart does not suppress them
    assert k([[0, 0], [2.0, 0], [1.0, 0]], [0.9, 0.8, 0.45]) == [True, True, False]


def properties():
    rng = np.random.RandomState(0)
    for trial in range(300):
        n = int(rng.randint(0, 60))
        pos = rng.uniform(-6, 6, (n, 2))
        e = rng.uniform(0, 1, n)                                                     # distinct values: the result does not depend on the order of the input
        R, thr = float(rng.choice([0.5, 1.0, 1.5, 2.0])), float(rng.choice([0.3, 0.5]))
        keep = RO.suppress_close_nodes(pos, e, R, thr)
        assert (keep == brute(pos, e, R, thr)).all(), (trial, R, thr)               # the independent reference
        assert (e[keep] >= thr).all()                                                # a subset of the nodes at or above the threshold
        kp = pos[keep]
        if len(kp) > 1:
            d = np.hypot(kp[:, None, 0] - kp[None, :, 0], kp[:, None, 1] - kp[None, :, 1]) + np.eye(len(kp)) * 1e9
            assert d.min() >= R                                                      # no two kept nodes closer than R
        for i in np.flatnonzero((e >= thr) & ~keep):                                  # maximal: every dropped candidate has a kept node closer than R with e at least its own
            dd = np.hypot(*(kp - pos[i]).T)
            assert (dd < R).any() and (e[keep][dd < R] >= e[i]).any()
        perm = rng.permutation(n)                                                     # the order of the input does not change the set
        assert (RO.suppress_close_nodes(pos[perm], e[perm], R, thr) == keep[perm]).all()
    # exact ties (a saturated sigmoid gives the same e more than once): the lower index goes first, as the independent reference says; 40 nodes with three distinct values
    for trial in range(100):
        n = 40
        pos = rng.uniform(-3, 3, (n, 2))
        e = rng.choice([0.9, 0.8, 0.5], n)
        keep = RO.suppress_close_nodes(pos, e, 1.5, 0.5)
        assert (keep == brute(pos, e, 1.5, 0.5)).all(), trial
        first_of_value = {v: np.flatnonzero(e == v)[0] for v in (0.9, 0.8, 0.5)}
        assert keep[first_of_value[0.9]]                                               # the first node of the highest value is always kept


def ties_and_dump():
    pos = np.array([[0, 0], [1.0, 0], [8, 8], [8.4, 8], [20, 20], [20.5, 20]], float)
    mk = lambda e: dict(nodes=dict(pos=pos, e=np.array(e, float)))  # noqa: E731
    sp = RO.parse("close:1.5@0.5")
    assert RO.tied_pairs([mk([0.9, 0.9, 0.7, 0.6, 0.9, 0.8])], sp) == 1                    # (0, 1) tie and are close; node 4 has the same e but is far from both
    assert RO.tied_pairs([mk([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])], sp) == 0                    # distinct values
    assert RO.tied_pairs([mk([0.4, 0.4, 0.7, 0.7, 0.3, 0.3])], sp) == 1                    # a tie below the threshold is not counted; (2, 3) is
    assert RO.tied_pairs([mk([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])], sp) == 3                    # three close pairs
    import pickle
    import tempfile
    g, _ = QE.toy_lane_gt()
    pred = qformat.oracle_prediction(g)
    pred["nodes"]["e"][3] = 0.1
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pred.pkl")
        assert RO.dump_nodes([pred, pred], [11, 12], path, e_min=0.2) == 2
        out = pickle.load(open(path, "rb"))
    M = len(pred["nodes"]["pos"])
    assert [o["row"] for o in out] == [11, 12] and len(out[0]["idx"]) == M - 1 and 3 not in out[0]["idx"].tolist()        # e below e_min is left out
    assert out[0]["pos"].shape == (M - 1, 2) and out[0]["flags"].shape[0] == M - 1 and out[0]["e"].dtype == np.float64
    assert (out[0]["pos"] == pred["nodes"]["pos"][out[0]["idx"]]).all()                                                   # the values are the model's, not rounded


def apply_and_spec():
    assert RO.parse("plain@0.4") == RO.Spec("plain", None, 0.4) and RO.parse("close:1.5@0.5") == RO.Spec("close", 1.5, 0.5) and RO.parse(" close:2@0.3 ") == RO.Spec("close", 2.0, 0.3)
    for bad in ("close@0.5", "close:1.5", "plain", "plain:1@0.5", "close:-1@0.5", "close:1.5@", "close:1.5@0", "close:1.5@0.0", "plain@0"):
        try:
            RO.parse(bad)
            raise SystemExit(f"{bad!r} accepted")
        except ValueError:
            pass
    assert RO.name(RO.parse("plain@0.5")) == "plain_t0.5" and RO.name(RO.parse("close:1.5@0.4")) == "close1.5_t0.4" and RO.name(RO.parse("close:1.25@0.5")) == "close1.25_t0.5"
    pred = dict(nodes=dict(pos=np.array([[0, 0], [1.0, 0], [4, 0]], float), e=np.array([[0.9], [0.8], [0.2]]), dirv=np.zeros((3, 2))), links=dict(p=np.ones(1)), assign=None)
    before = copy.deepcopy(pred)
    assert RO.apply(pred, RO.parse("plain@0.5")) is pred
    q = RO.apply(pred, RO.parse("close:1.5@0.5"))
    assert q is not pred and q["links"] is pred["links"] and q["nodes"]["pos"] is pred["nodes"]["pos"]      # the arrays and the links are shared
    assert q["nodes"]["e"].shape == (3, 1) and q["nodes"]["e"].reshape(-1).tolist() == [0.9, 0.0, 0.0]      # dropped and sub-threshold nodes are at 0, the shape stays
    assert (pred["nodes"]["e"] == before["nodes"]["e"]).all()                                                 # the original is untouched
    assert RO.counts([pred, pred], RO.parse("close:1.5@0.5")) == (4, 2) and RO.counts([pred], RO.parse("plain@0.5")) == (2, 2)
    assert RO.timed([pred] * 500, RO.parse("close:1.5@0.5")) < 1.0                                            # ms PER frame (the total of 500 frames of three nodes is several ms): the division by the frames
    e_thr = dict(nodes=dict(pos=np.array([[0, 0], [5, 5]], float), e=np.array([0.5, 0.5])))
    assert RO.counts([e_thr], RO.parse("close:1.5@0.5")) == (2, 2) and RO.counts([e_thr], RO.parse("plain@0.5")) == (2, 2)   # e equal to the threshold counts as delivered, before and after


def with_duplicate(pred, i, shift, e, link=True):
    """a copy of the oracle prediction with one more node: `shift` m from node i, existence e, the section and assignment of node i, and (link) one link from it to the successor of i"""
    q = copy.deepcopy(pred)
    nd, lk, a = q["nodes"], q["links"], q["assign"]
    M = len(nd["pos"])
    for key in ("pos", "dirv", "flags", "s", "cell", "section"):
        nd[key] = np.concatenate([nd[key], nd[key][i:i + 1]])
    nd["pos"][M] = nd["pos"][i] + np.asarray(shift, float)
    nd["e"] = np.concatenate([nd["e"], [e]])
    a["P"] = np.concatenate([a["P"], a["P"][i:i + 1]])
    a["road"] = np.concatenate([a["road"], a["road"][i:i + 1]])
    if link:
        succ = [int(d) for s, d in zip(lk["src"], lk["dst"]) if s == i]
        assert succ, "node has no successor"
        lk["src"] = np.concatenate([lk["src"], [M]])
        lk["dst"] = np.concatenate([lk["dst"], [succ[0]]])
        lk["p"] = np.concatenate([lk["p"], [1.0]])
        lk["long"] = np.concatenate([lk["long"], [False]])
        lk["a"] = np.concatenate([lk["a"], [1 / 3]])
        lk["b"] = np.concatenate([lk["b"], [1 / 3]])
        lk["curves"] = list(lk["curves"]) + [None]
    return q


def scorer():
    g, old = QE.toy_lane_gt()
    pred = qformat.oracle_prediction(g)
    M = len(pred["nodes"]["pos"])
    L0 = len(QE.kept(pred)[1])
    ref = QE.node_metrics(pred, g)
    assert ref["node_P"] == 1.0 and ref["node_R"] == 1.0
    i = 2                                                                                   # an interior node of the toy lane (it has a successor)
    dup = with_duplicate(pred, i, [0.4, 0.0], 0.6)                                          # a duplicate node 0.4 m from node i, weaker than node i (e = 1)
    nd = QE.node_metrics(dup, g)
    assert nd["node_P"] < 1.0 and len(QE.kept(dup)[1]) == L0 + 1                            # the duplicate counts as a false node and its link is read out
    spec = RO.parse("close:1.5@0.5")
    # what the step does to the ORACLE itself: the toy lane has real nodes only 1.0 m apart (the stop node at the split, 1 m after a regular node; the window node at 40 m): the lower index
    # stays, the other is dropped. That is the recall ceiling of the step (prediction 3 of the card), here 2 of 42 nodes
    base = RO.apply(pred, spec)
    lost = np.flatnonzero(base["nodes"]["e"] == 0.0).tolist()
    assert lost == [20, 40], lost
    out = RO.apply(dup, spec)
    assert out["nodes"]["e"][M] == 0.0 and (out["nodes"]["e"][:M] == base["nodes"]["e"]).all()   # the duplicate goes, and nothing else changes against the oracle with the step
    nn, nb = QE.node_metrics(out, g), QE.node_metrics(base, g)
    assert nn["node_P"] == nb["node_P"] == 1.0 and nn["node_R"] == nb["node_R"] < 1.0           # precision is back at 1.0; recall is the oracle's with the step
    assert len(QE.kept(out)[1]) == len(QE.kept(base)[1]) < L0                                   # the link of the duplicate is gone: no link to a dropped node
    m1, mb = QE.graph_metrics(QE.readout(out)[0], old), QE.graph_metrics(QE.readout(base)[0], old)
    assert all(abs(m1[k] - mb[k]) < 1e-9 for k in ("F1@1.0", "topo_F1", "lanes_found", "junction_F1", "dup_frac")), (m1, mb)    # the graph is that of the oracle with the step
    assert QE.assign_metrics(out, g)["lane_acc"] == QE.assign_metrics(base, g)["lane_acc"]
    # a duplicate that is stronger than the real node wins: the real node goes (the rule only looks at e)
    strong = with_duplicate(pred, i, [0.4, 0.0], 1.0, link=False)
    strong["nodes"]["e"][i] = 0.9
    o2 = RO.apply(strong, RO.parse("close:1.5@0.5"))
    assert o2["nodes"]["e"][i] == 0.0 and o2["nodes"]["e"][M] == 1.0
    # two nodes of the oracle 0.5 m apart are both true and stay only when the radius is below their distance (the recall ceiling of the step: prediction 3 of the card)
    close = copy.deepcopy(pred)
    close["nodes"]["pos"][1] = close["nodes"]["pos"][0] + [0.5, 0.0]
    assert RO.apply(close, RO.parse("close:0.4@0.5"))["nodes"]["e"][1] == 1.0 and RO.apply(close, RO.parse("close:1.5@0.5"))["nodes"]["e"][1] == 0.0
    return M


def exempt_and_mnms():
    """the special-type exemption (`x`, `x<p>`) and the Matrix-NMS spec (E288 / E296): a flagged node survives the close-node rule and the Matrix-NMS decay; the spec names are stable; `plain` with `x` and sigma <= 0 are refused"""
    s = RO.parse("close:1.5x@0.5")
    assert s.exempt and s.kind == "close" and RO.name(s) == "close1.5x_t0.5", s
    s2 = RO.parse("mnms:1.75s0.1x@0.5")
    assert s2.exempt and s2.sigma == 0.1 and RO.name(s2) == "mnms1.75s0.1x_t0.5"
    assert not RO.parse("close:1.5@0.5").exempt
    pos = np.array([[0, 0], [0.4, 0], [5, 5]], float)
    e = np.array([0.95, 0.9, 0.8])
    flags = np.zeros((3, 8))
    flags[1, 1] = 0.9                                                                   # node 1 is an exit
    pred = dict(nodes=dict(pos=pos, e=e, flags=flags))
    assert RO.apply(pred, RO.parse("close:1.5@0.5"))["nodes"]["e"][1] == 0.0
    assert RO.apply(pred, RO.parse("close:1.5x@0.5"))["nodes"]["e"][1] == 0.9
    assert RO.apply(pred, RO.parse("mnms:1.75s0.1@0.5"))["nodes"]["e"][1] == 0.0
    assert RO.apply(pred, RO.parse("mnms:1.75s0.1x@0.5"))["nodes"]["e"][1] == 0.9
    sp = RO.parse("close:1.5x0.95@0.5")
    assert sp.exempt == 0.95 and RO.name(sp) == "close1.5x0.95_t0.5", sp
    assert RO.apply(pred, sp)["nodes"]["e"][1] == 0.0                                 # the exit flag 0.9 is below 0.95: not protected
    for bad in ("plain:0.5x", "mnms:1.75s0@0.5"):
        try:
            RO.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"spec {bad!r} must be refused")


def main():
    exempt_and_mnms()
    cases()
    properties()
    ties_and_dump()
    apply_and_spec()
    M = scorer()
    print(f"qgraph_readout: rule cases, 300 random frames against an independent reference, order / threshold / spacing properties, apply and spec parsing, scorer on the toy lane with a duplicate "
          f"node ({M} oracle nodes): ok")


if __name__ == "__main__":
    main()

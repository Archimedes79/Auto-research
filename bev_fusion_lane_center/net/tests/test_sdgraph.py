"""E390 tests of the SD road-graph encoding (`net/prep/lane_sdgraph_data.py`, `net/model/jsdgraph.py`, `evaluation/sdgraph_probe.py`); CPU, hand graphs, no cache needed (the VAL cache part is skipped when absent).

    wsl bash tools/env/_py14.sh net/tests/test_sdgraph.py

leak (critic B1): every relation and degree comes from the INPUT graph: a dropped link has no succ / pred / sibling relation and does not count in the degrees, and a scorer that reads only the pair's relations scores
AUC 0.5 on the dropped links against the non-links; the textual mutant that builds the relations from the exact graph scores 1.0 (caught); the deciding P2 set leaves out the pairs whose nodes both carry a route flag.
invariance: features and embedding unchanged under a rigid motion; a mutant node feature with the absolute ego x breaks it (caught). bayes: at error 0 the input; on a chain of equal counts it corrects a flipped
count; the noise matrix by hand. errors: clipping, unknown, hide, the smooth field keeps joints joined, Douglas-Peucker. probe metrics: AUC / AP by hand.
"""
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in ("../model", "../prep", "../loss", "../../training", "../../training/data", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jsdgraph as JS  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402
import sdgraph_probe as SP  # noqa: E402

torch.set_num_threads(2)


def mutant(mod, *edits):
    """the module's source with each (old, new) replaced (old found exactly once), loaded as its own module"""
    src = open(mod.__file__, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    m = types.ModuleType(mod.__name__ + "_mutant")
    m.__file__ = mod.__file__
    exec(compile(src, mod.__file__, "exec"), m.__dict__)
    return m


def chain_graph(n=8, lanes=1):
    """n straight roads in a chain along y (each 20 m, linked end to start), all with the same lane class, the route along all of them"""
    geo = [[np.stack([np.zeros(21), np.linspace(20 * i, 20 * i + 20, 21) - 80.0], 1)] for i in range(n)]
    return dict(kind=np.zeros(n, int), geo=geo, lanes=np.full(n, lanes), route=np.arange(n), succ=[(i, i + 1) for i in range(n - 1)], oppl=[])


def twin_chains(n=10):
    """a chain of n roads along +y and its opposite chain along -y, 3 m to the side (the negatives of P2: near pairs without a link), no route"""
    a = chain_graph(n)
    geo = a["geo"] + [[np.stack([np.full(21, 3.0), (np.linspace(20 * i, 20 * i + 20, 21) - 80.0)[::-1]], 1)] for i in range(n)]
    succ = a["succ"] + [(n + i + 1, n + i) for i in range(n - 1)]
    return dict(kind=np.zeros(2 * n, int), geo=geo, lanes=np.zeros(2 * n, int), route=np.full(2 * n, -1), succ=succ, oppl=[(i, n + i) for i in range(n)] + [(n + i, i) for i in range(n)])


def test_leak():
    G = twin_chains(10)
    scores, labs = [], []
    for s in range(30):
        Gin, T = SG.perturb(G, np.random.default_rng(s), drop=0.5)
        f = SG.featurize(Gin)
        tg = SP.targets(G, Gin, T, f["pair"])
        for a, b in T["dropped"]:
            assert not f["rel"][0, a, b] and not f["rel"][1, b, a] and not f["rel"][3, a, b], "a dropped link left a relation"
        sel = tg["dec"]
        rel_any = f["rel"][[0, 1, 3]].any(0)                                    # the link-type relations (succ, pred, sibling)
        scores.append(rel_any[sel].astype(float))
        labs.append(tg["link"][sel])
        deg_out = f["x"][:, 9] * 4
        for a, b in T["dropped"]:
            assert deg_out[a] == sum(1 for x, _ in Gin["succ"] if x == a), "the out-degree counts a dropped link"
    auc = SP.auc(np.concatenate(scores), np.concatenate(labs))
    assert auc <= 0.55, f"the input relations give the dropped links away: AUC {auc}"
    M = mutant(SG, ("Rm = np.zeros((len(REL), N, N), bool)\n    for a, b in G[\"succ\"]:", "Rm = np.zeros((len(REL), N, N), bool)\n    for a, b in G.get(\"succ_true\", G[\"succ\"]):"),
               ("Gin = dict(G, geo=geo, lanes_in=lanes_in,", "Gin = dict(G, succ_true=G[\"succ\"], geo=geo, lanes_in=lanes_in,"))
    scores, labs = [], []
    for s in range(30):
        Gin, T = M.perturb(G, np.random.default_rng(s), drop=0.5)
        f = M.featurize(Gin)
        tg = SP.targets(G, Gin, T, f["pair"])
        scores.append(f["rel"][[0, 1, 3]].any(0)[tg["dec"]].astype(float))
        labs.append(tg["link"][tg["dec"]])
    assert SP.auc(np.concatenate(scores), np.concatenate(labs)) > 0.95, "the leak mutant was not caught"
    # route order: pairs whose nodes both carry a route flag are not deciding
    Gr = chain_graph(6)
    Gin, T = SG.perturb(Gr, np.random.default_rng(0), drop=0.5)
    tg = SP.targets(Gr, Gin, T, SG.pair_features(Gin))
    flag = Gin["route_in"] >= 0
    assert not (tg["dec"] & flag[:, None] & flag[None, :]).any()
    print(f"leak: dropped links leave no relation and no degree; link-relation-only scorer AUC {auc:.3f} (<= 0.55), the exact-graph mutant caught; route-flagged pairs not deciding: ok")


def test_invariance():
    G = SG.toy()
    a, _ = SG.perturb(G, np.random.default_rng(0))
    b, _ = SG.perturb(G, np.random.default_rng(0), rigid=(9.0, 4.0, -0.5))
    fa, fb = SG.featurize(a), SG.featurize(b)
    assert np.allclose(fa["x"], fb["x"], atol=1e-5) and np.allclose(fa["pair"], fb["pair"], atol=1e-5)
    M = mutant(SG, ("row += [ext / 20.0, deg_in[i] / 4.0, deg_out[i] / 4.0, near[i] / 10.0, float(rin[i] >= 0), rin[i] / 20.0 if rin[i] >= 0 else -0.05, 0.0, 0.0]",
                    "row += [ext / 20.0, deg_in[i] / 4.0, deg_out[i] / 4.0, near[i] / 10.0, float(rin[i] >= 0), rin[i] / 20.0 if rin[i] >= 0 else -0.05, float(pts[0, 0]) / 50.0, 0.0]"))
    ma, mb = M.node_features(a), M.node_features(b)
    assert not np.allclose(ma, mb, atol=1e-5), "the absolute-pose mutant was not caught"
    enc = JS.SDGraphEncoder(SG.NODE_F, SG.EDGE_F).eval()
    ba, bb = JS.collate([fa]), JS.collate([fb])
    with torch.no_grad():
        ha = enc(ba["x"], ba["rel"], ba["pair"].float(), ba["valid"])
        hb = enc(bb["x"], bb["rel"], bb["pair"].float(), bb["valid"])
    assert (ha - hb).abs().max() <= 1e-4 * ha.abs().mean()
    print("invariance: features and embedding unchanged under a rigid motion, the absolute-pose mutant caught: ok")


def test_bayes():
    O = SP.noise_matrix(0.3)
    assert np.allclose(O.sum(1), 1) and O[0, 0] == 0.7 and O[0, 1] == 0.3 and O[2, 1] == 0.15 and O[2, 3] == 0.15
    G = chain_graph(8, lanes=2)
    fit = SP.bayes_fit([G] * 20)
    Gin, _ = SG.perturb(G, np.random.default_rng(0))
    assert (SP.bayes_p1(Gin, fit, 0.0) == G["lanes"]).all()
    Gin = dict(Gin, lanes_in=Gin["lanes_in"].copy())
    Gin["lanes_in"][3] = 1                                                    # one flipped count inside a chain of class 2
    assert SP.bayes_p1(Gin, fit, 0.3)[3] == 2, "the neighbours do not correct a single flip"
    print("bayes: the noise matrix by hand, the input at error 0, a single flip in a consistent chain corrected: ok")


def test_errors():
    G = SG.toy()
    Gc, _ = SG.perturb(G, np.random.default_rng(2), count_rate=1.0)
    road = G["kind"] == 0
    assert ((Gc["lanes_in"][road] - G["lanes"][road]) != 0).all() and Gc["lanes_in"][road].min() >= 0 and Gc["lanes_in"][road].max() <= SG.N_LANE - 1
    Gu, _ = SG.perturb(G, np.random.default_rng(2), unknown_p=1.0)
    assert (Gu["lanes_in"][road] == SG.N_LANE).all() and (Gu["lanes_in"][~road] == G["lanes"][~road]).all()
    Gh, Th = SG.perturb(G, np.random.default_rng(2), hide=1.0)
    assert (Gh["route_in"] == -1).all() and set(Th["hidden"].tolist()) == set(np.flatnonzero(G["route"] >= 0).tolist())
    f = SG.smooth_field(np.random.default_rng(5), 2.0)
    p = np.array([[3.0, 4.0], [3.0, 4.0]])
    assert np.allclose(f(p)[0], f(p)[1])
    Gs, _ = SG.perturb(G, np.random.default_rng(5), sigma=2.0)
    assert np.linalg.norm(Gs["geo"][0][0][-1] - Gs["geo"][1][0][0]) - np.linalg.norm(G["geo"][0][0][-1] - G["geo"][1][0][0]) < 0.5, "the smooth field opens a joint"
    assert len(SG.rdp(np.stack([np.linspace(0, 10, 11), np.zeros(11)], 1), 0.1)) == 2
    print("errors: count flips clipped, unknown, hide, the smooth field keeps joints, Douglas-Peucker: ok")


def test_metrics():
    assert SP.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75 and SP.auc([1, 1], [0, 1]) == 0.5
    assert abs(SP.ap([0.9, 0.8, 0.1], [1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-12
    r = SP.Rows()
    r.p1 = {0: (np.array([1, 2]), np.array([1, 1])), 1: (np.array([0]), np.array([0]))}
    assert r.metric("p1") == 2 / 3 and r.metric("p1", [1]) == 1.0
    print("metrics: AUC, AP, pooled P1 by hand: ok")


def lane_gt_frame(location, two_lane):
    """a hand lane-GT frame driving +y: in left-hand traffic (Singapore) the divider is on the RIGHT; road r1 has its divider lane at x = 3.5 (and, when two_lane, a kerb lane at x = 0); road r2 = the
    opposite direction at x = 7 driving -y"""
    y = np.linspace(-40, 40, 161)
    def piece(pid, x, road, left, right, opp, idx, cnt, rev=False):
        pts = np.stack([np.full_like(y, x), y[::-1] if rev else y], 1)
        return dict(id=pid, pts=pts, succ=[], pred=[], length=80.0, lane_count=cnt, left=left, right=right, opposite=opp, road=road, lane_index=idx, tier=np.zeros(len(y), int))
    ps = [piece("d", 3.5, "r1", ["k"] if two_lane else [], [], ["o"], 1 if two_lane else 0, 2 if two_lane else 1)]
    if two_lane:
        ps.append(piece("k", 0.0, "r1", [], ["d"], [], 0, 2))
    ps.append(piece("o", 7.0, "r2", [], [], ["d"], 0, 1, rev=True))
    roads = [dict(id="r1", kind="road", pieces=[p["id"] for p in ps if p["road"] == "r1"], succ=[], opposite=["r2"]), dict(id="r2", kind="road", pieces=["o"], succ=[], opposite=["r1"])]
    return dict(location=location, pieces=ps, roads=roads)


def test_divider():
    """critic B2: the road line is the divider-side edge: a 2-lane and a 1-lane road that share the divider lane give the SAME line (left-hand traffic: the divider is on the right); the mutant that takes the
    LEFTMOST lane (the kerb lane in Singapore) is caught"""
    G2 = SG.frame_graph(lane_gt_frame("singapore-onenorth", True), route=[])
    G1 = SG.frame_graph(lane_gt_frame("singapore-onenorth", False), route=[])
    l2, l1 = G2["geo"][0][0], G1["geo"][0][0]
    assert G2["lanes"][0] == 1 and G1["lanes"][0] == 0 and np.allclose(l2, l1) and np.allclose(l2[:, 0], 3.5 + SG.HALF_LANE), (l2[:3], l1[:3])
    M = mutant(SG, ('    side = "left" if rh else "right"                                          # the divider side of a road', '    side = "left"'))
    m2 = M.frame_graph(lane_gt_frame("singapore-onenorth", True), route=[])["geo"][0][0]
    m1 = M.frame_graph(lane_gt_frame("singapore-onenorth", False), route=[])["geo"][0][0]
    assert not np.allclose(m2, m1), "the leftmost-lane mutant was not caught"
    print("divider: a 2-lane and a 1-lane road with the same divider lane give the same line (left-hand traffic), the leftmost-lane mutant caught: ok")


def test_paired_levels():
    """the VAL levels share one draw per frame: the flipped counts of count error 0.1 are a subset of those of 0.3, the dropped links of drop 0.1 a subset of those of 0.3"""
    G = twin_chains(10)
    a, ta = SP.val_draw(G, 3, "ce01")
    b, tb = SP.val_draw(G, 3, "base")
    fa, fb = a["lanes_in"] != G["lanes"], b["lanes_in"] != G["lanes"]
    assert (fa <= fb).all() and set(SP.val_draw(G, 3, "d01")[1]["dropped"]) <= set(tb["dropped"])
    c, _ = SP.val_draw(G, 3, "s2")
    assert (c["lanes_in"] == b["lanes_in"]).all() and c["succ"] == b["succ"], "the deformation level changed the count / drop draws"
    print("paired levels: nested count flips and drops, the deformation drawn last: ok")


def test_p6():
    """D135 (P6): the proxy-segment features are invariant under a joint rigid transform of map and proxies, a map-only pose error changes them; an absent segment is masked in the batch, its links are gone and
    its proxies' target is 'unassigned' (-1); the nearest-segment rule assigns the toy's proxies (1 m beside their line) on the exact map"""
    G = SG.toy()
    a, Ta = SG.perturb(G, np.random.default_rng(0))
    b, _ = SG.perturb(G, np.random.default_rng(0), rigid=(6.0, -2.0, 0.7))
    pa, pb = SG.prox_features(a), SG.prox_features(b)
    assert pa.shape == (len(G["prox"]), 5, SG.PA_F) and np.allclose(pa, pb, atol=1e-5), "the proxy features are not invariant under a joint transform"
    c, _ = SG.perturb(G, np.random.default_rng(0), pose=2.0)
    assert not np.allclose(SG.prox_features(c), pa, atol=1e-3), "a map-only pose error must move the map against the proxies"
    assert (SP.p6_rule(a, pa, 2.0) == Ta["pseg"]).mean() > 0.9, (SP.p6_rule(a, pa, 2.0), Ta["pseg"])
    d, Td = SG.perturb(G, np.random.default_rng(1), node_drop=1.0)
    assert (Td["pseg"] == -1).all() and d["succ"] == [] and d["oppl"] == [] and not d["present"].any()
    e, Te = SG.perturb(G, np.random.default_rng(4), node_drop=0.5)
    gone = ~e["present"]
    assert gone.any() and e["present"].any()
    assert ((Te["pseg"] == -1) == gone[G["pseg"]]).all(), "the proxies of a missing segment must be unassigned, the others keep their segment"
    import jsdgraph as JSm
    fe = SG.featurize(e)
    bt = JSm.collate([fe])
    assert (bt["valid"][0] == torch.as_tensor(e["present"])).all() and (fe["pa"][:, gone] == 0).all() and not fe["rel"][:, gone].any()
    print("p6: proxy features invariant under a joint transform and moved by a map pose error; a missing segment masked, its links gone, its proxies unassigned; the nearest rule on the exact toy: ok")


def test_cut():
    """critic of v3 (B1): a piece that runs from road r1 into intersection x1 (listed on both) gives the intersection only its stretch INSIDE x1 and the road only its stretch outside; its proxies are left out
    (two right answers); a piece listed on one node keeps its proxies with that node as the target"""
    y = np.linspace(0, 40, 81)
    pts = np.stack([np.zeros_like(y), y], 1)
    p1 = dict(id="a", pts=pts, succ=[], pred=[], length=40.0, lane_count=1, left=[], right=[], opposite=[], road="r1", lane_index=0, tier=np.zeros(81, int),
              segments=[dict(token="l", kind="lane", i0=0, i1=59, in_intersection=False), dict(token="c", kind="connector", i0=60, i1=80, in_intersection=True)])
    q = np.stack([np.linspace(-20, -2, 37), np.full(37, 60.0)], 1)
    p2 = dict(id="b", pts=q, succ=[], pred=[], length=18.0, lane_count=1, left=[], right=[], opposite=[], road="r2", lane_index=0, tier=np.zeros(37, int),
              segments=[dict(token="m", kind="lane", i0=0, i1=36, in_intersection=False)])
    g = dict(location="boston-seaport", pieces=[p1, p2], roads=[dict(id="r1", kind="road", pieces=["a"], succ=["x1"], opposite=[]), dict(id="x1", kind="intersection", pieces=["a"], succ=[], opposite=[]),
                                                                dict(id="r2", kind="road", pieces=["b"], succ=[], opposite=[])])
    G = SG.frame_graph(g, route=[])
    road_line, inter = G["geo"][0][0], G["geo"][1]
    assert len(inter) == 1 and inter[0][:, 1].min() >= 30.0 - 1e-6 and len(inter[0]) == 21, "the intersection holds more than its inner stretch"
    assert road_line[:, 1].max() <= 29.5 + 1e-6, "the road line runs into the intersection"
    assert set(G["pseg"].tolist()) == {2} and G["prox_pieces"] == (1, 2), (G["pseg"], G["prox_pieces"])
    print("cut: the intersection holds only its inner stretch, the road only its outer one, a piece on two nodes gives no proxy: ok")


def test_two_intersections():
    """critic of v4 (B1): a piece road -> x1 -> road -> x2 is listed on both intersections: each gets only the inner stretch inside ITS area; the nearest rule answers 'unassigned' when no present segment is
    aligned, also at tau = inf (critic of v4, B2)"""
    y = np.linspace(0, 80, 161)
    pts = np.stack([np.zeros_like(y), y], 1)
    segs = [dict(token="l1", kind="lane", i0=0, i1=39, in_intersection=False), dict(token="c1", kind="connector", i0=40, i1=59, in_intersection=True),
            dict(token="l2", kind="lane", i0=60, i1=99, in_intersection=False), dict(token="c2", kind="connector", i0=100, i1=119, in_intersection=True),
            dict(token="l3", kind="lane", i0=120, i1=160, in_intersection=False)]
    p = dict(id="a", pts=pts, succ=[], pred=[], length=80.0, lane_count=1, left=[], right=[], opposite=[], road="r1", lane_index=0, tier=np.zeros(161, int), segments=segs)
    sq = lambda y0, y1: [[-5.0, y0], [5.0, y0], [5.0, y1], [-5.0, y1]]  # noqa: E731
    g = dict(location="boston-seaport", pieces=[p], areas=dict(intersection=[dict(id="x1", exterior=sq(19.5, 30.0)), dict(id="x2", exterior=sq(49.5, 60.0))]),
             roads=[dict(id="r1", kind="road", pieces=["a"], succ=["x1"], opposite=[]), dict(id="x1", kind="intersection", pieces=["a"], succ=[], opposite=[]),
                    dict(id="x2", kind="intersection", pieces=["a"], succ=[], opposite=[])])
    G = SG.frame_graph(g, route=[])
    x1, x2 = G["geo"][1], G["geo"][2]
    assert len(x1) == 1 and len(x2) == 1 and x1[0][:, 1].max() <= 30.0 and x2[0][:, 1].min() >= 49.5, "an intersection got the other one's inner stretch"
    pa = np.zeros((2, 3, SG.PA_F), np.float32)
    pa[..., 7] = 1.0
    pa[..., 1] = -1.0                                                         # no aligned segment
    assert (SP.p6_rule(None, pa, np.inf) == -1).all(), "the nearest rule assigned a proxy without an aligned segment"
    print("two intersections: each holds only its own inner stretch; the nearest rule at tau = inf: unassigned without an aligned segment: ok")


def test_invariance_real():
    """card v6: on the first 150 VAL frames (skipped without the cache) every encoder input is unchanged under joint rigid transforms (largest change <= 1e-6, no relation or present flip); the two P4 defects of
    the v5 read, re-introduced as mutants, are caught: (a) a node without geometry kept present (its fallback point at the ego origin), (b) the attach choice by plain argmin (exact ties decided by float noise)"""
    try:
        va = SG.load("val")
    except FileNotFoundError as e:                                            # a missing cache skips; a stale one (AssertionError of `load`) fails
        print(f"invariance_real: skipped ({e})")
        return
    graphs = va["graphs"][:150]
    ok = SP.feature_invariance(graphs, draws=2)
    assert max(ok["x"], ok["pair"], ok["pa"]) <= 1e-6 and ok["rel_flips"] == 0 and ok["present_flips"] == 0, ok
    ma = mutant(SG, ('    present = (rng.random(N) >= node_drop) & np.array([any(len(p) >= 2 for p in gs) for gs in G["geo"]], bool)',
                     '    present = (rng.random(N) >= node_drop) & np.ones(N, bool)'),
                ('    present &= np.array([any(len(p) >= 2 and np.ptp(p, 0).max() > 1e-6 for p in gs) for gs in geo], bool)',
                 '    present &= True'))
    mb = mutant(SG, ("            i1, i2 = first_min(d1), first_min(d2)\n            if d1[i1] <= d2[i2] + TIE_TOL:",
                     "            i1, i2 = np.unravel_index(int(np.argmin(d1)), d1.shape), np.unravel_index(int(np.argmin(d2)), d2.shape)\n            if d1[i1] <= d2[i2]:"))
    ra, rb = SP.feature_invariance(graphs, draws=2, mod=ma), SP.feature_invariance(graphs, draws=2, mod=mb)
    assert max(ra["x"], ra["pair"], ra["pa"]) > 1e-3 or ra["rel_flips"] > 0, ("the empty-node mutant was not caught", ra)
    assert rb["pair"] > 1e-3, ("the plain-argmin mutant was not caught", rb)
    print(f"invariance_real: 150 VAL frames x 2 transforms, largest input change {max(ok['x'], ok['pair'], ok['pa']):.1e}, 0 flips; the empty-node mutant (pair {ra['pair']:.3f}, rel flips {ra['rel_flips']}) and the "
          f"plain-argmin mutant (pair {rb['pair']:.3f}) caught: ok")


def test_stated_rate():
    """card v7: `featurize(stated_rate=True)` differs from `featurize()` only in column STATED_F of the present roads (a mutant writing another slot, e.g. the route order, is caught); the 7 arms set the flag in
    `train` and `load_models`; the misstated report levels change only the stated rate, not the draw"""
    G = SG.toy()
    a, _ = SG.perturb(G, np.random.default_rng(0), count_rate=0.3)
    x0, x1 = SG.featurize(a)["x"], SG.featurize(a, stated_rate=True)["x"]
    diff = np.flatnonzero(np.abs(x0 - x1).max(0) > 0)
    assert diff.tolist() == [SG.STATED_F] and np.allclose(x1[G["kind"] == 0, SG.STATED_F], 0.3)
    M = mutant(SG, ("STATED_F = 13 ", "STATED_F = 12 "))
    m1 = M.featurize(a, stated_rate=True)["x"]
    assert np.flatnonzero(np.abs(x0 - m1).max(0) > 0).tolist() != [13], "the slot mutant was not caught"
    import inspect
    assert "set_arm(arm)" in inspect.getsource(SP.train) and "set_arm(arm)" in inspect.getsource(SP.load_models), "the 7 arms must set the stated-rate flag in train and load_models"
    g1, t1 = SP.val_draw(G, 2, "mis_lo")
    g2, t2 = SP.val_draw(G, 2, "base")
    assert (g1["lanes_in"] == g2["lanes_in"]).all() and g1["count_rate"] == 0.0 and g2["count_rate"] == 0.3
    print("stated rate: only column 13 changes (the slot mutant caught), set_arm in train / load_models, the misstated level changes only the stated rate: ok")


def test_selftests():
    SG.selftest()
    JS.selftest()


TESTS = [test_leak, test_invariance, test_bayes, test_errors, test_metrics, test_divider, test_paired_levels, test_p6, test_cut, test_two_intersections, test_invariance_real, test_stated_rate, test_selftests]
if __name__ == "__main__":
    for t in TESTS:
        t()
    print("test_sdgraph ok")

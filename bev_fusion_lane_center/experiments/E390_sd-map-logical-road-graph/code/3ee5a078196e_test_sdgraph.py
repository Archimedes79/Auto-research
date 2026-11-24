"""E390 tests of the SD road-graph encoding (`training/data/lane_sdgraph_data.py`, `training/grid/jsdgraph.py`, `evaluation/sdgraph_probe.py`); CPU, hand graphs, no cache needed (the VAL cache part is skipped when absent).

    wsl bash tools/env/_py14.sh training/grid/test_sdgraph.py

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
for sub_ in (".", "../data", "../../evaluation"):
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


def test_selftests():
    SG.selftest()
    JS.selftest()


TESTS = [test_leak, test_invariance, test_bayes, test_errors, test_metrics, test_selftests]
if __name__ == "__main__":
    for t in TESTS:
        t()
    print("test_sdgraph ok")

"""E394 tests of the map-error catalogue (`training/data/lane_maperror_data.py`) and the step-0 probe's metrics (`evaluation/maperror_probe.py`); CPU, the hand graph of E390 (`lane_sdgraph_data.toy`), no cache needed.

    wsl bash tools/env/_py14.sh training/grid/test_maperror.py

Per mode: what changes and what must not (the labels, the other nodes); the draws are NESTED (a higher level only adds to a lower one: ghosts, spurious links, swapped links, count errors, missing intersections),
deterministic, and independent across modes (adding one mode never changes the draw of another or of E390's `perturb`); every error leaves a graph the encoder's features can be built from; wrong_scene removes
every true segment (T['present'] all False); the wrong-map label; the TRAIN mix has about a third wrong maps and every spec of it is valid. A handful of textual mutants of the module must each fail a test.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "../data", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import lane_maperror_data as ME  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402

N_DRAWS = 40


def big():
    """the toy graph twice, the second copy 60 m to the side (10 nodes, 6 links: enough elements for the rate tests)"""
    G = SG.toy()
    n = len(G["kind"])
    geo2 = [[p + np.array([60.0, 0.0]) for p in gs] for gs in G["geo"]]
    return dict(kind=np.concatenate([G["kind"]] * 2), geo=G["geo"] + geo2, lanes=np.concatenate([G["lanes"]] * 2), route=np.concatenate([G["route"], np.full(n, -1)]),
                succ=G["succ"] + [(a + n, b + n) for a, b in G["succ"]], oppl=G["oppl"] + [(a + n, b + n) for a, b in G["oppl"]], prox=G["prox"], pdir=G["pdir"], pseg=G["pseg"])


def mutant(mod, *edits):
    src = open(mod.__file__, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    m = types.ModuleType(mod.__name__ + "_mutant")
    m.__file__ = mod.__file__
    exec(compile(src, mod.__file__, "exec"), m.__dict__)
    return m


def test_modes(M=ME):
    G = big()
    n = len(G["kind"])
    # no error = E390's perturb of the same generator
    Gin, T = M.apply(G, np.random.default_rng(0), {})
    assert (Gin["lanes_in"] == G["lanes"]).all() and Gin["succ"] == G["succ"] and T["present"].all() and T["ghosts"] == 0 and len(Gin["kind"]) == n
    # shift: every map point moves by exactly the shift, the labels and the links do not
    Gs, Ts = M.apply(G, np.random.default_rng(1), dict(shift=5.0))
    d = np.concatenate([np.linalg.norm(a - b, axis=1) for gs, hs in zip(Gs["geo"], G["geo"]) for a, b in zip(gs, hs)])
    assert np.allclose(d, 5.0) and Gs["succ"] == G["succ"] and Ts["present"].all(), "shift"
    # yaw: a rotation about the ego: lengths from the origin kept, the angle moved by yaw
    Gy, _ = M.apply(G, np.random.default_rng(2), dict(yaw=3.0))
    a, b = np.concatenate(G["geo"][0]), np.concatenate(Gy["geo"][0])
    assert np.allclose(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    ang = np.arctan2(b[:, 1], b[:, 0]) - np.arctan2(a[:, 1], a[:, 0])
    assert np.allclose(np.abs(np.rad2deg(np.angle(np.exp(1j * ang)))), 3.0, atol=1e-6), "yaw"
    # count_big: every road's known class moves by two (clipped) at rate 1; unknown stays unknown
    Gc, _ = M.apply(G, np.random.default_rng(3), dict(count_big=1.0))
    roads = G["kind"] == 0
    assert (Gc["lanes_in"][roads] != G["lanes"][roads]).all(), "count_big"
    # ghost: extra nodes appended, the original nodes and the labels untouched, the nodes never a target
    Gg, Tg = M.apply(G, np.random.default_rng(4), dict(ghost=0.6))
    k = Tg["ghosts"]
    assert k >= 1 and len(Gg["kind"]) == n + k and len(Tg["present"]) == n and (Gg["kind"][:n] == G["kind"]).all() and (Gg["route_in"][n:] == -1).all() and Gg["present"][n:].all() and (Gg["kind"][n:] == 0).all()
    assert all(len(gs[0]) >= 2 for gs in Gg["geo"][n:]), "a ghost road has a line"
    # link_extra: new links are new, within the gap, the old ones stay
    Ge, _ = M.apply(G, np.random.default_rng(5), dict(link_extra=1.0))
    new = [e for e in Ge["succ"] if e not in G["succ"]]
    assert set(G["succ"]) <= set(Ge["succ"]) and len(new) >= 1 and all(e[0] != e[1] for e in new), "link_extra"
    assert len(Ge["succ"]) == len(set(Ge["succ"])) and len(Ge["succ"]) - len(G["succ"]) == len(new), "a spurious link repeats an existing link"
    for a_, b_ in new:
        ex, en = SG.ends(Ge, a_)[2], SG.ends(Ge, b_)[0]
        assert np.linalg.norm(ex[:, None] - en[None], axis=-1).min() <= M.LINK_GAP + 1e-9
    # link_swap: out- and in-degrees are kept, the set changes
    Gw, _ = M.apply(G, np.random.default_rng(6), dict(link_swap=1.0))
    assert sorted(a for a, _ in Gw["succ"]) == sorted(a for a, _ in G["succ"]) and sorted(b for _, b in Gw["succ"]) == sorted(b for _, b in G["succ"]) and set(Gw["succ"]) != set(G["succ"]), "link_swap"
    # outdated at 1: no intersection left, its approach roads linked to its exits, the labels say 'missing'
    Go, To = M.apply(G, np.random.default_rng(7), dict(outdated=1.0))
    inter = np.flatnonzero(G["kind"] == 1)
    assert (~Go["present"][inter]).all() and (~To["present"][inter]).all() and (0, 2) in Go["succ"] and (0, 3) in Go["succ"] and (5, 7) in Go["succ"], "outdated"
    assert all(Go["present"][a] and Go["present"][b] for a, b in Go["succ"])
    # wrong_route: another path flagged, the true one not
    Gr, _ = M.apply(G, np.random.default_rng(8), dict(wrong_route=1))
    assert set(np.flatnonzero(Gr["route_in"] >= 0)) != set(np.flatnonzero(G["route"] >= 0)) and (Gr["route_in"] >= 0).any(), "wrong_route"
    # wrong_scene: the other graph, every true segment missing
    H = SG.toy()
    Gx, Tx = M.apply(G, np.random.default_rng(9), dict(wrong_scene=1), other=H)
    assert len(Gx["kind"]) == len(H["kind"]) and not Tx["present"].any() and len(Tx["present"]) == n, "wrong_scene"
    for spec in (dict(shift=5.0), dict(yaw=3.0), dict(count_big=1.0), dict(ghost=0.6), dict(link_extra=1.0), dict(link_swap=1.0), dict(outdated=1.0), dict(wrong_route=1), dict(wrong_scene=1),
                 dict(pose=2.0, count_rate=0.4, drop=0.4, hide=1.0, node_drop=0.4, eps=2.0, sigma=2.0, unknown_p=0.6)):
        f = SG.featurize(M.apply(G, np.random.default_rng(10), spec, other=H)[0])
        assert np.isfinite(f["x"]).all() and np.isfinite(f["pair"]).all(), spec
    try:
        M.apply(G, np.random.default_rng(0), dict(bogus=1))
        raise SystemExit("an unknown key was accepted")
    except AssertionError:
        pass
    try:
        M.apply(G, np.random.default_rng(0), dict(wrong_scene=1))
        raise SystemExit("wrong_scene without another graph was accepted")
    except AssertionError:
        pass


def test_nested_and_independent(M=ME):
    G = big()
    for key, lo, hi in (("ghost", 0.1, 0.6), ("link_extra", 0.2, 1.0), ("link_swap", 0.2, 1.0), ("count_big", 0.3, 1.0), ("outdated", 0.3, 1.0)):
        for s in range(N_DRAWS):
            a, ta = M.apply(G, np.random.default_rng(s), {key: lo})
            b, tb = M.apply(G, np.random.default_rng(s), {key: hi})
            if key == "ghost":
                assert tb["ghosts"] >= ta["ghosts"] and all(np.array_equal(x[0], y[0]) for x, y in zip(a["geo"][len(G["kind"]):], b["geo"][len(G["kind"]):])), "ghosts are not nested"
            elif key == "link_extra":
                assert set(a["succ"]) <= set(b["succ"]), "spurious links are not nested"
            elif key == "link_swap":
                assert len(set(a["succ"]) ^ set(G["succ"])) <= len(set(b["succ"]) ^ set(G["succ"])), "swaps are not nested"
            elif key == "count_big":
                assert ((a["lanes_in"] != G["lanes"]) <= (b["lanes_in"] != G["lanes"])).all(), "count errors are not nested"
            else:
                assert ((~a["present"]) <= (~b["present"])).all(), "missing intersections are not nested"
    # the same generator state and spec give the same map; a new mode does not move another mode's draw or E390's
    a, _ = M.apply(G, np.random.default_rng(5), dict(shift=3.0, count_rate=0.4, drop=0.4))
    b, _ = M.apply(G, np.random.default_rng(5), dict(shift=3.0, count_rate=0.4, drop=0.4))
    c, _ = M.apply(G, np.random.default_rng(5), dict(shift=3.0, count_rate=0.4, drop=0.4, ghost=0.6, link_swap=0.3, yaw=2.0))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a["geo"], b["geo"])) and a["succ"] == b["succ"], "not deterministic"
    assert (a["lanes_in"] == c["lanes_in"][: len(a["lanes_in"])]).all() and (a["present"] == c["present"][: len(a["present"])]).all(), "an added mode changed the draws of E390's modes"
    for s in range(10):                                        # the smooth deformation draws extra numbers in `perturb`: the new modes' generators must not move (they are drawn before it)
        w0, _ = M.apply(G, np.random.default_rng(s), dict(wrong_route=1, link_swap=0.5))
        w1, _ = M.apply(G, np.random.default_rng(s), dict(wrong_route=1, link_swap=0.5, sigma=2.0))
        assert (w0["route_in"] == w1["route_in"]).all() and w0["succ"] == w1["succ"], "a base mode moved the draw of a new mode"
    p, _ = M.apply(G, np.random.default_rng(5), dict(shift=3.0))
    q, _ = M.apply(G, np.random.default_rng(5), dict(shift=3.0, ghost=0.6))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(p["geo"], q["geo"][: len(p["geo"])])), "the ghost draw moved the shift draw"


def test_mix_and_labels(M=ME):
    rng = np.random.default_rng(0)
    specs = [M.train_spec(rng) for _ in range(4000)]
    share = np.mean([M.is_wrong(s) for s in specs])
    assert 0.30 <= share <= 0.40, f"the TRAIN mix has {share:.2f} wrong maps: the card says about a third"
    assert abs(np.mean([s == {} for s in specs]) - 0.12) < 0.03
    G, H = big(), SG.toy()
    for s in specs[:300]:
        assert set(s) <= set(M.BASE_KEYS) | set(M.NEW_KEYS)
        M.apply(G, np.random.default_rng(1), s, other=H)
    assert M.is_wrong(dict(wrong_scene=1)) and M.is_wrong(dict(shift=4.0)) and M.is_wrong(dict(wrong_route=1)) and M.is_wrong(dict(yaw=5.0)) and not M.is_wrong(dict(shift=2.0, count_rate=0.4)) and not M.is_wrong({})
    sc = np.array([0, 0, 1, 1, 2, 2])
    pl = M.other_plan(sc, np.array([5, 5, 5, 2, 5, 5]))
    assert all(sc[pl[i]] != sc[i] for i in range(6)) and pl[0] == 4 or pl[0] != 3, pl
    assert all(np.array([5, 5, 5, 2, 5, 5])[pl] >= 3), "the other map has at least 3 nodes"
    assert [c["mode"] for c in M.CATALOGUE][:3] == ["pose", "shift", "yaw"] and all(c["key"] in set(M.BASE_KEYS) | set(M.NEW_KEYS) for c in M.CATALOGUE)
    assert M.DECIDING["right"] == dict(pose=1.0, count_rate=0.1, drop=0.1, hide=0.3, node_drop=0.1) and M.DECIDING["wrong"] == dict(wrong_scene=1)


def test_mutants():
    edits = [
        ("shift is not applied", ('t = float(spec["shift"]) * np.array([np.cos(ang), np.sin(ang)])', "t = 0.0 * np.array([np.cos(ang), np.sin(ang)])")),
        ("ghosts are not extended", ('        _extend(Gin, kind, geo, lanes)\n    return len(kind)', "        pass\n    return len(kind)")),
        ("a ghost gets a route flag", ('np.full(m, -1, int)])\n    Gin["present"]', 'np.full(m, 0, int)])\n    Gin["present"]')),
        ("spurious links may exist already", ("if a != b and Gin[\"present\"][a] and Gin[\"present\"][b] and (a, b) not in have:", 'if a != b and Gin["present"][a] and Gin["present"][b]:')),
        ("swapped links change the degrees", ("S[i], S[j] = (a, d), (c, b)", "S[i], S[j] = (a, d), (c, d)")),
        ("a missing intersection stays labelled present", ('T["present"] = T["present"] & ~gone[: len(T["present"])]', 'T["present"] = T["present"]')),
        ("wrong_scene keeps the labels", ('T["present"] = np.zeros(len(T["present"]), bool)', 'T["present"] = T["present"]')),
        ("the per-mode generators are drawn after perturb", ("sub = [np.random.default_rng(int(s)) for s in rng.integers(1 << 62, size=len(NEW_KEYS))]       # one generator per new mode, drawn BEFORE perturb: E390's draws are what they were\n    g = dict(zip(NEW_KEYS, sub))\n    Gin, T = SG.perturb(G, rng, **{k: v for k, v in spec.items() if k in BASE_KEYS})",
                                                            "Gin, T = SG.perturb(G, rng, **{k: v for k, v in spec.items() if k in BASE_KEYS})\n    sub = [np.random.default_rng(int(s)) for s in rng.integers(1 << 62, size=len(NEW_KEYS))]\n    g = dict(zip(NEW_KEYS, sub))")),
        ("the wrong scene is not in the mix", ("        return {**simple, \"wrong_scene\": 1}", "        return dict(simple)")),
        ("a wrong shift is 8 m", ("WRONG_SHIFT, WRONG_YAW = 4.0, 4.0", "WRONG_SHIFT, WRONG_YAW = 8.0, 4.0")),
    ]
    dead = 0
    for name, (old, new) in edits:
        M = mutant(ME, (old, new))
        try:
            test_modes(M)
            test_nested_and_independent(M)
            test_mix_and_labels(M)
        except (AssertionError, SystemExit, IndexError, KeyError, ValueError, TypeError):
            dead += 1
            continue
        raise SystemExit(f"the mutant '{name}' survived")
    print(f"maperror mutants: {dead} of {len(edits)} killed")


def test_probe_metrics():
    import maperror_probe as MPR
    import mapassoc_probe as MP
    r = MP.Rows()
    r.a = {0: (np.array([1, -1, 2, 3]), np.array([1, -1, -1, 4]), np.array([0, 1, 2, 0]))}
    e = MPR.extra_metrics(r, [0])
    assert abs(e["wrong_assign"] - 0.5) < 1e-9 and abs(e["assign_rate"] - 0.75) < 1e-9, e
    sp = dict(MPR.spec_list(["all"]))
    assert sp["exact"] == {} and sp["shift=4.0"] == {"shift": 4.0} and sp["wrong_scene=1"] == {"wrong_scene": 1} and sp["dec"]["pose"] == 1.0 and sp["wrong_scene+dec"]["wrong_scene"] == 1 and sp["wrong_scene+dec"]["pose"] == 1.0
    L = MPR.Lall()
    b = dict(kept=__import__("torch").as_tensor(np.array([[True, True, False, True]])), lanes_all=__import__("torch").as_tensor(np.array([[1, 2, 3, -1]])))
    L.add(0, b, 0, np.array([1, 0, 3, 3]))
    assert abs(L.acc([0]) - 0.5) < 1e-9, "the lane read-out counts the delivered tokens with a true class only"


if __name__ == "__main__":
    ME.selftest()
    test_modes()
    test_nested_and_independent()
    test_mix_and_labels()
    test_probe_metrics()
    test_mutants()
    print("test_maperror ok")

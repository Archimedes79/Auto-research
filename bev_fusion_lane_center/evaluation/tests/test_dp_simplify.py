"""Spec tests of the output stage simplify_dp (E257, decision D20) in training/pipeline/p16_infer.py, on toy graphs (no network, no GPU).
Usage (gcar): bash tools/env/_py14.sh test_dp_simplify.py    (pytest also collects the test_ functions)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_infer as PI  # noqa: E402


def graph(polys, ids, edges=()):
    return dict(pts=[np.asarray(p, float) for p in polys], edges=list(edges), nodes=[list(i) for i in ids])


def dist_to_polyline(p, q):
    d = np.inf
    for a, b in zip(q[:-1], q[1:]):
        ab = b - a
        s = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = min(d, float(np.linalg.norm(p - (a + s * ab))))
    return d


def test_D1_straight_run_keeps_only_its_ends():
    g = graph([np.c_[np.linspace(0, 50, 8), np.zeros(8)]], [range(8)])
    s = PI.simplify_dp(g, 0.2)
    assert len(s["pts"][0]) == 2 and s["nodes"][0] == [0, 7]
    assert np.allclose(s["pts"][0], g["pts"][0][[0, 7]])


def test_D2_a_bend_beyond_the_tolerance_is_kept_and_a_small_one_is_not():
    corner = np.array([[0, 0], [10, 0], [20, 0], [20, 10], [20, 20.0]])
    s = PI.simplify_dp(graph([corner], [range(5)]), 0.2)
    assert s["nodes"][0] == [0, 2, 4]  # the corner stays, the straight interior points go
    wobble = np.array([[0, 0], [10, 0.1], [20, 0], [30, -0.1], [40, 0.0]])
    s = PI.simplify_dp(graph([wobble], [range(5)]), 0.2)
    assert s["nodes"][0] == [0, 4]
    s = PI.simplify_dp(graph([wobble], [range(5)]), 0.05)
    assert len(s["nodes"][0]) > 2  # the same wobble is kept at a tighter tolerance


def test_D3_shared_nodes_and_ends_survive_even_when_collinear():
    a = np.c_[np.linspace(0, 40, 5), np.zeros(5)]  # nodes 10..14, node 12 is a junction used by b
    b = np.c_[np.linspace(20, 20, 4), np.linspace(0, 15, 4)]  # starts at node 12
    g = graph([a, b], [[10, 11, 12, 13, 14], [12, 20, 21, 22]], edges=[(0, 1)])
    s = PI.simplify_dp(g, 0.5)
    assert s["nodes"][0] == [10, 12, 14] and s["nodes"][1] == [12, 22]
    assert s["edges"] == [(0, 1)]
    ends_before = {(i, nd[0], nd[-1]) for i, nd in enumerate(g["nodes"])}
    ends_after = {(i, nd[0], nd[-1]) for i, nd in enumerate(s["nodes"])}
    assert ends_before == ends_after


def test_D4_no_removed_vertex_is_farther_than_eps_from_the_new_polyline():
    rng = np.random.default_rng(1)
    for eps in (0.1, 0.3, 1.0):
        x = np.linspace(0, 80, 60)
        q = np.c_[x, 6 * np.sin(x / 15) + rng.normal(0, 0.05, len(x))]
        s = PI.simplify_dp(graph([q], [range(60)]), eps)
        worst = max(dist_to_polyline(p, s["pts"][0]) for p in q)
        assert worst <= eps + 1e-9, (eps, worst)
        assert len(s["pts"][0]) < len(q)


def test_D5_idempotent_and_node_ids_are_a_subset_in_order():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 60, 40)
    q = np.c_[x, 3 * np.sin(x / 9)]
    g = graph([q, q[:10] + [0, 4]], [list(range(40)), list(range(100, 110))])
    s1 = PI.simplify_dp(g, 0.3)
    s2 = PI.simplify_dp(s1, 0.3)
    assert all(np.array_equal(a, b) for a, b in zip(s1["pts"], s2["pts"])) and s1["nodes"] == s2["nodes"]
    for old, new in zip(g["nodes"], s1["nodes"]):
        assert new == [i for i in old if i in set(new)]  # subsequence of the original ids


def test_D6_short_polylines_pass_unchanged():
    g = graph([[[0, 0], [5, 0]], [[0, 1.0]]], [[1, 2], [3]])
    s = PI.simplify_dp(g, 0.5)
    assert s["nodes"] == [[1, 2], [3]]


def test_D7_a_forked_graph_from_graph_from_vertices_keeps_edges_ends_and_shared_ids():
    import p16_common as C

    V = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [30, 6], [40, 12.0]])
    E = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]  # a fork at vertex 2
    g = C.graph_from_vertices(V, E)
    s = PI.simplify_dp(g, 0.2)
    assert s["edges"] == list(g["edges"]) and len(s["pts"]) == len(g["pts"])
    shared = {i for i in {i for nd in g["nodes"] for i in nd} if sum(i in nd for nd in g["nodes"]) > 1}
    kept = {i for nd in s["nodes"] for i in nd}
    assert shared <= kept
    for old, new in zip(g["nodes"], s["nodes"]):
        assert new[0] == old[0] and new[-1] == old[-1]


def test_D8_assemble_refuses_dp_with_aux_and_negative_eps_before_any_work():
    for kw in (dict(dp_eps=0.2, aux=True), dict(dp_eps=-1.0)):
        try:
            PI.assemble(None, None, 0.7, None, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")


def test_D9_inconsistent_input_is_refused():
    for g in (dict(pts=[np.zeros((3, 2))], edges=[], nodes=None), dict(pts=[np.zeros((3, 2)), np.zeros((3, 2))], edges=[], nodes=[[1, 2, 3]]),
              dict(pts=[np.zeros((4, 2))], edges=[], nodes=[[1, 2, 3]])):
        try:
            PI.simplify_dp(g, 0.2)
        except AssertionError:
            continue
        raise AssertionError("inconsistent graph accepted")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails} / {len(tests)} passed")
    sys.exit(1 if fails else 0)

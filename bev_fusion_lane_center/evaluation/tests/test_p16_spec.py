"""Spec tests T01 - T16, T21, T22 of the verified-ray hypothesis tree, run against the numpy reference p16_ref with an ORACLE verification (exact tolerances).
The network version has to pass the same scenes later with statistical tolerances (see Hypothesis_Tree_Spec_and_Tests_EN.md).
Usage (gcar): python test_p16_spec.py [-v]    (pytest also collects the test_ functions)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_ref as R  # noqa: E402

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- toy scenes
def line(x0, x1, y=0.0, n=300):
    x = np.linspace(x0, x1, n)
    return np.c_[x, y + 0 * x]


def arc(radius, length=80.0, n=600):
    s = np.linspace(0, length, n)
    if radius >= 1e8:
        return np.c_[s, 0 * s]
    return np.c_[radius * np.sin(s / radius), radius * (1 - np.cos(s / radius))]


def incoming(n=11):
    return np.c_[np.linspace(-10, 0, n), np.zeros(n)]


def ray_line(angle_deg, length=60.0, n=300):
    a = math.radians(angle_deg)
    s = np.linspace(0, length, n)
    return np.c_[s * math.cos(a), s * math.sin(a)]


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def toy_road():
    x1, x2, x3 = np.linspace(0, 25, 120), np.linspace(25, 70, 240), np.linspace(0, 48, 240)
    main = np.c_[x1, 0 * x1]
    S = np.c_[x2, 0 * x2]
    U = np.c_[x2, 14 * smooth((x2 - 25) / 45)]
    M = np.c_[x3, -14 * (1 - smooth(x3 / 48))]
    return [np.r_[main, S], np.r_[main, U], M]


def alive_children(ch):
    return [c for c in ch if c["alive"]]


# ---------------------------------------------------------------- level 0: geometry, no scene
def test_T01_pattern_counts_and_envelope():
    rays = R.make_rays((0, 0), 0.0)
    assert len(rays) == 12
    assert sum(len(r["radii"]) for r in rays) == 28
    by = {r["phi"]: r for r in rays}
    assert abs(by[0]["r"] - 24.0) < 1e-9 and abs(by[20]["r"] - 12.0) < 1e-9 and abs(by[-20]["r"] - 12.0) < 1e-9
    assert abs(by[9]["r"] - 24.0 / (1 + (9 / 20) ** 2)) < 1e-9
    for a in (50, -50, 90, -90, 125, -125, 180):
        assert abs(by[a]["r"] - 4.0) < 1e-9
    rs = [by[a]["r"] for a in (0, 9, 20, 50, 90, 125, 180)]
    assert all(x >= y - 1e-9 for x, y in zip(rs, rs[1:]))  # longer rays at smaller angles
    for r in rays:
        assert all((abs(x - rk) < 1e-9 for rk in R.CFG["rings"]) or abs(x - r["r"]) < 1e-9 for x in r["radii"])
        assert np.allclose(np.linalg.norm(r["pts"], axis=1), r["radii"], atol=1e-9)


def test_T02_equivariance_and_mirror():
    c, th = np.array([3.1, -2.2]), 0.7
    r0, r1 = R.make_rays((0, 0), 0.0), R.make_rays(c, th)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    for a, b in zip(r0, r1):
        assert np.allclose(a["pts"] @ rot.T + c, b["pts"], atol=1e-9)
    by = {r["phi"]: r for r in r0}
    for a in (9, 20, 50, 90, 125):
        assert np.allclose(by[a]["pts"] * np.array([1, -1]), by[-a]["pts"], atol=1e-9)


def test_T03_survival_algebra():
    for K in range(1, 9):
        for _ in range(50):
            v = RNG.uniform(0.01, 0.99, K)
            T, P = R.survival(v)
            assert abs(P.sum() - 1.0) < 1e-12 and (P >= -1e-15).all() and (np.diff(T) <= 1e-15).all() and len(P) == K + 1
    T, P = R.survival(np.ones(4))
    assert abs(P[-1] - 1.0) < 1e-12
    T, P = R.survival(np.array([0.0, 0.5, 0.5]))
    assert abs(P[0] - 1.0) < 1e-12


# ---------------------------------------------------------------- level 1: one tip, oracle verification
def test_T04_straight_lane_single_hypothesis():
    sc = R.Scene([line(-10, 80)])
    ch, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2)
    al = alive_children(ch)
    best = max(ch, key=lambda c: c["nu"])
    assert len(al) == 1 and len(ch) == 2
    assert np.linalg.norm(best["pos"] - np.array([24.0, 0.0])) < 0.5 and best["nu"] >= 0.9


def test_T05_step_length_grows_with_radius():
    steps = []
    for rad in (10, 15, 30, 100, 1e9):
        sc = R.Scene([arc(rad)])
        ch, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2)
        al = alive_children(ch)
        assert al, f"no living child at R={rad}"
        best = max(al, key=lambda c: c["nu"])
        assert float(sc.dist(best["pos"])[0]) < 0.8  # the vertex lies on the lane
        steps.append(float(np.linalg.norm(best["pos"])))
    assert all(b >= a - 0.5 for a, b in zip(steps, steps[1:])), steps  # non-decreasing
    assert steps[0] <= 8.5 and abs(steps[-1] - 24.0) < 0.5, steps


def test_T06_neighbouring_lane_is_not_a_continuation():
    sc = R.Scene([line(-10, 80, 0.0), line(-10, 80, -3.5)])
    ch, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2)
    for c in alive_children(ch):
        assert abs(c["pos"][1]) < 1.0, c


def test_T07_fork_gives_two_hypotheses_only_if_resolvable():
    def n_alive(angle):
        sc = R.Scene([line(-10, 80), ray_line(angle)])
        ch, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2)
        return len(alive_children(ch)), ch, sc

    n, ch, sc = n_alive(18)
    assert n == 2
    S, U = R.Scene([line(-10, 80)]), R.Scene([ray_line(18)])
    ends = [c["pos"] for c in alive_children(ch)]
    near_S = [e for e in ends if S.dist(e)[0] < 0.7 and U.dist(e)[0] > 1.5]
    near_U = [e for e in ends if U.dist(e)[0] < 0.7 and S.dist(e)[0] > 1.5]
    assert len(near_S) == 1 and len(near_U) == 1
    assert n_alive(12)[0] == 2 and n_alive(6)[0] == 2  # resolved down to about 6 deg (rays at +-9 deg, sigma 0.6 m)
    assert n_alive(3)[0] == 1 and n_alive(0)[0] == 1  # below about 5 deg the branch is the same lane
    sc1 = R.Scene([line(-10, 80)])
    assert len(alive_children(R.expand(sc1, (0, 0), 0.0, incoming(), b=2)[0])) == 1  # no fork, no second living child, slot still present


def test_T08_coverage_and_backward_rays():
    sc = R.Scene([line(-60, 80)])
    com = incoming()
    ch, cands = R.expand(sc, (0, 0), 0.0, com, b=2)
    back = [c for c in cands if abs(c["ray"]["phi"]) == 180][0]
    assert back["nu"] < 0.3
    for c in ch:
        if not c["dummy"]:
            assert np.min(np.linalg.norm(com - c["pos"], axis=1)) >= R.CFG["cov_r"] - 1e-9


def test_T09_redundancy_and_order_invariance():
    def cand(x, y, nu):
        return dict(end=np.array([x, y]), nu=nu)

    cs = [cand(24, 0, 0.95), cand(4, 0.6, 0.94), cand(3.8, 1.4, 0.6), cand(2.8, 2.6, 0.5)]
    tip = np.zeros(2)
    kept, _ = R.select(cs, tip, 2)
    ends = {tuple(np.round(c["end"], 2)) for c in kept}
    assert ends == {(24.0, 0.0), (3.8, 1.4)}, ends  # the stub on the long segment is dominated, the near duplicate endpoint is suppressed; of two collinear candidates the longer wins
    for _ in range(10):
        p = RNG.permutation(len(cs))
        k2, _ = R.select([cs[i] for i in p], tip, 2)
        assert {tuple(np.round(c["end"], 2)) for c in k2} == ends


def test_T10_lane_end_stops_the_tree():
    sc = R.Scene([line(-10, 10)])
    nodes = R.grow_tree(sc, (0, 0), 0.0, D=3, b=2, committed0=incoming())
    xs = [n["pos"][0] for n in nodes if n["alive"]]
    assert max(xs) <= 10.5 and max(xs) >= 7.0, xs


def test_T11_gap_strict_versus_tolerant():
    sc = R.Scene([line(-10, 10), line(13, 80)])
    strict, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2)
    tol, _ = R.expand(sc, (0, 0), 0.0, incoming(), b=2, tolerant=True)
    bs, bt = max(strict, key=lambda c: c["nu"]), max(tol, key=lambda c: c["nu"])
    assert bs["pos"][0] <= 9.0
    assert bt["pos"][0] >= 23.0 and bt["nu"] < bs["nu"]  # crosses the gap, at a penalty


def test_T12_seed_without_heading_gives_two_lobes():
    sc = R.Scene([line(-60, 60)])
    ch, _ = R.expand(sc, (0, 0), 0.0, [], b=2, bipolar=True)
    al = alive_children(ch)
    assert len(al) == 2
    xs = sorted(c["pos"][0] for c in al)
    assert abs(xs[0] + 24) < 0.5 and abs(xs[1] - 24) < 0.5
    assert abs(abs(math.degrees(R.wrap(al[0]["th"] - al[1]["th"]))) - 180) < 3


# ---------------------------------------------------------------- level 2: tree
def test_T13_fixed_shape_and_locality():
    sc = R.Scene([line(-10, 80), ray_line(18)])
    nodes = R.grow_tree(sc, (0, 0), 0.0, D=3, b=2, committed0=incoming())
    assert R.level_counts(nodes) == [1, 2, 4, 8]
    for i, n in enumerate(nodes[1:], 1):
        assert n["parent"] < i and nodes[n["parent"]]["level"] == n["level"] - 1
        if n["dummy"]:
            assert not n["alive"]
    far = R.Scene([line(-10, 80), ray_line(18), line(-10, 80, 40.0)])  # a lane 40 m away must not change anything
    n2 = R.grow_tree(far, (0, 0), 0.0, D=3, b=2, committed0=incoming())
    assert all(np.allclose(a["pos"], b["pos"]) and a["alive"] == b["alive"] for a, b in zip(nodes, n2))


def forest():
    sc = R.Scene(toy_road())
    roots = [((0.0, 0.0), 0.0), ((0.0, -14.0), 0.0)]
    nodes = []
    for p, th in roots:
        t = R.grow_tree(sc, p, th, D=8, b=2, committed0=[(p[0] - k, p[1]) for k in (2, 4, 6)])
        off = len(nodes)
        for n in t:
            n = dict(n)
            n["parent"] = n["parent"] + off if n["parent"] >= 0 else -1
            nodes.append(n)
    return sc, nodes


def graph_metrics(sc, pos, edges):
    pts = []
    for a, b in edges:
        k = max(int(np.linalg.norm(pos[b] - pos[a]) / 0.5), 1)
        t = np.linspace(0, 1, k + 1)[:, None]
        pts.append(pos[a] + t * (pos[b] - pos[a]))
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    gt = np.concatenate([np.c_[np.linspace(l[0, 0], l[-1, 0], 200), np.interp(np.linspace(l[0, 0], l[-1, 0], 200), l[:, 0], l[:, 1])] for l in sc.lanes])
    from scipy.spatial import cKDTree
    pt = cKDTree(np.r_[pts, pos]) if len(pts) else cKDTree(pos)
    recall = float(np.mean(pt.query(gt)[0] < 1.0))
    false = float(np.mean(sc.dist(pts) > 1.0)) if len(pts) else 1.0
    return recall, false


def test_T14_toy_road_becomes_a_dag_with_fork_and_merge():
    sc, nodes = forest()
    pos, edges = R.merge_nodes(nodes)
    recall, false = graph_metrics(sc, pos, edges)
    assert recall >= 0.85 and false <= 0.10, (recall, false)
    outd, ind = {}, {}
    for a, b in edges:
        outd[a] = outd.get(a, 0) + 1
        ind[b] = ind.get(b, 0) + 1
    # fork and merge vertices are resolved to the verification tolerance: the lanes are one lane while they are within about 1 m (fork 25 m -> branch 1 m apart at 32 m; merge 48 m, lanes within 1 m from 40 m)
    fork = [i for i, d in outd.items() if d >= 2 and 18.0 <= pos[i][0] <= 36.0 and abs(pos[i][1]) < 3.0]
    merge = [i for i, d in ind.items() if d >= 2 and 38.0 <= pos[i][0] <= 50.0 and abs(pos[i][1]) < 2.0]
    assert fork and merge, (fork, merge)


def test_T15_no_duplicate_vertices_after_merge():
    sc, nodes = forest()
    pos, edges = R.merge_nodes(nodes)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + 10 * np.eye(len(pos))
    assert d.min() >= 1.0, d.min()


def test_T16_beam_budget_keeps_the_best():
    sc, nodes = forest()
    keep = R.prune_top_m(nodes, 5)
    for lv in {n["level"] for n in nodes}:
        idx = [i for i, n in enumerate(nodes) if n["level"] == lv and n["alive"]]
        kept = [i for i in idx if i in keep]
        rest = [i for i in idx if i not in keep]
        assert len(kept) <= 5
        if kept and rest:
            assert min(nodes[i]["pi"] for i in kept) >= max(nodes[i]["pi"] for i in rest) - 1e-12


# ---------------------------------------------------------------- level 4: time
def test_T21_recursion_raises_true_and_lowers_false_nodes():
    pi = np.array([0.5, 0.5])
    hist = [pi.copy()]
    for _ in range(5):
        pi = R.fuse(pi, np.array([0.9, 0.05]))
        hist.append(pi.copy())
    h = np.array(hist)
    assert (np.diff(h[:, 0]) > 0).all() and (np.diff(h[:, 1]) < 0).all()
    assert h[-1, 0] > 0.9 and h[-1, 1] < 0.1, h[-1]


def test_T22_ego_warp_is_rigid_and_invertible():
    xy = RNG.uniform(-30, 30, (20, 2))
    th = RNG.uniform(-math.pi, math.pi, 20)
    dx, dy, dpsi = 3.0, 0.4, 0.1
    q, th2 = R.se2_apply(xy, th, dx, dy, dpsi)
    c, s = math.cos(dpsi), math.sin(dpsi)
    back = np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]] + np.array([dx, dy])
    assert np.allclose(back, xy, atol=1e-9)
    assert np.allclose(np.linalg.norm(q[:, None] - q[None], axis=-1), np.linalg.norm(xy[:, None] - xy[None], axis=-1), atol=1e-9)
    assert np.allclose(th2, th - dpsi)
    q0, _ = R.se2_apply(xy, th, 0, 0, 0)
    assert np.allclose(q0, xy)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_T")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", name, "->", repr(e)[:300])
            if verbose:
                import traceback
                traceback.print_exc()
    print(f"{len(tests) - fails} / {len(tests)} passed")
    sys.exit(1 if fails else 0)

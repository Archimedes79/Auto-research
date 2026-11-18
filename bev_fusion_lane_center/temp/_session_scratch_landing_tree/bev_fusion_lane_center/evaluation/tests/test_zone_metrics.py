"""E282 test of `zone_metrics` (CPU): known-answer cases on the toy lane (perfect head, two nodes per zone, empty head, a stray node, the neighbour cell, a dense node that is in no zone), the radius,
additivity of the pooling, the paired bootstrap, the matched threshold, and the density against `qgraph_eval` (the E276 convention of nodes per 100 m).

    wsl bash tools/env/_py14.sh test_zone_metrics.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../../training/grid"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402
import qzones as QZ  # noqa: E402
import zone_metrics as ZM  # noqa: E402
from test_tiers_sparse import toy  # noqa: E402

SPARSE = (0.3, 7.5)
LEN = 100.0                                                                        # a GT lane length for the tests that do not look at the density


def setup(sparse=SPARSE):
    g = toy(labelled=False)
    zs = QZ.zones(g, sparse)
    return g, zs, zs["nd"]["pos"][zs["gt_node"]]


def counts(zs, P, ln=LEN):
    return ZM.frame_counts(zs, ln, P)


def test_perfect_head():
    g, zs, Psp = setup()
    Z, nf = zs["n_zones"], int((zs["kind"] == 1).sum())
    c = counts(zs, Psp)
    assert (c["n_free"], c["n_fixed"]) == (nf, Z - nf) and c["n_deliv"] == Z and c["n_off"] == 0
    assert (c["free_rec"], c["free_exact"]) == (nf, nf) and (c["fixed_rec"], c["fixed_exact"]) == (Z - nf, Z - nf)
    p = ZM.pool([c])
    assert all(p[k] == 1.0 for k in ("free_zone_recall", "free_zone_exact", "fixed_zone_recall", "fixed_zone_exact", "zone_recall", "zone_exact")) and p["off_zone"] == 0.0
    assert abs(p["nodes_per_100m"] - 100.0 * Z / LEN) < 1e-12
    print("test_perfect_head ok: zones", Z, "free", nf)


def test_two_nodes_per_zone():
    g, zs, Psp = setup()
    free = zs["kind"] == 1
    nf, nx = int(free.sum()), int((~free).sum())
    P2 = Psp + 0.5 * zs["nd"]["dirv"][zs["gt_node"]]                                 # a second node 0.5 m along the lane, inside the zone's dense stretch
    c = counts(zs, np.concatenate([Psp, P2[free]]))
    assert c["free_rec"] == nf and c["free_exact"] == 0, "two nodes in every free zone: recall 1, exact 0"
    assert c["fixed_rec"] == c["fixed_exact"] == nx and c["n_off"] == 0
    p = ZM.pool([c])
    assert p["free_zone_recall"] == 1.0 and p["free_zone_exact"] == 0.0 and p["off_zone"] == 0.0
    c = counts(zs, np.concatenate([Psp, P2[~free]]))                                 # and the same for the fixed zones alone
    assert c["fixed_rec"] == nx and c["fixed_exact"] == 0 and c["free_exact"] == nf and c["n_off"] == 0
    p = ZM.pool([counts(zs, np.concatenate([Psp, P2]))])
    assert p["zone_recall"] == 1.0 and p["zone_exact"] == 0.0 and p["fixed_zone_exact"] == 0.0
    print("test_two_nodes_per_zone ok")


def test_empty_head():
    g, zs, _ = setup()
    c = counts(zs, np.zeros((0, 2)))
    assert c["n_deliv"] == 0 and c["n_off"] == 0 and c["free_rec"] == c["free_exact"] == c["fixed_rec"] == c["fixed_exact"] == 0
    p = ZM.pool([c])
    assert p["zone_recall"] == 0.0 and p["zone_exact"] == 0.0 and p["free_zone_recall"] == 0.0 and p["nodes_per_100m"] == 0.0
    assert p["off_zone"] != p["off_zone"], "no delivered node: off_zone is NaN (0 / 0), not an error"
    q = ZM.pool([])
    assert all(v != v for v in q.values()), "no frames: everything NaN"
    print("test_empty_head ok")


def test_stray_node_and_radius():
    g, zs, Psp = setup()
    z = zs["zone_of"]
    # a node 10 m from every lane, and one outside the box of the metrics (counted, as qgraph_eval counts it in `dens`)
    c = counts(zs, np.concatenate([Psp, [[-10.0, 0.0], [200.0, 200.0]]]))
    assert c["n_off"] == 2 and c["n_deliv"] == len(Psp) + 2 and c["dens"][0] == len(Psp) + 2
    assert c["free_exact"] == c["n_free"] and c["fixed_exact"] == c["n_fixed"], "a stray node changes no zone"
    assert abs(ZM.pool([c])["off_zone"] - 2.0 / (len(Psp) + 2)) < 1e-12
    # the 2.0 m tolerance: 1.9 m and exactly 2.0 m from the dense node of a free zone are in, 2.3 m is off (lateral to the lane x = 0, zone 1 of the toy)
    m = int(zs["gt_node"][1])
    x0, y0 = zs["nd"]["pos"][m]
    assert z[m] == 1 and zs["kind"][1] == 1 and x0 == 0.0
    for dx, off in ((-1.9, 0), (-2.0, 0), (-2.3, 1)):
        assert counts(zs, [[x0 + dx, y0]])["n_off"] == off, dx
    print("test_stray_node_and_radius ok")


def test_neighbour_cell_counts_by_position():
    g, zs, Psp = setup()
    nd = zs["nd"]
    dense_cells = {tuple(c) for c in nd["cell"]}
    gts = [int(m) for z, m in enumerate(zs["gt_node"]) if zs["kind"][z] == 1 and nd["piece"][m] in (0, 1)]       # the free zones on the straight lane x = 0 (pieces A and B of the toy)
    assert len(gts) >= 10
    P = nd["pos"][gts] + [-1.5, 0.0]                                                 # 1.5 m beside the zone's GT node: the cell next door, which holds no dense GT node
    cells = QT.cell_of(P)
    assert all(tuple(c) not in dense_cells for c in cells) and (cells != nd["cell"][gts]).any(axis=1).all(), "the trap: the slot cell is no GT cell"
    c = counts(zs, P)
    assert c["n_off"] == 0 and c["free_rec"] == len(gts) and c["free_exact"] == len(gts), "by position all of them are in their zone"
    print("test_neighbour_cell_counts_by_position ok:", len(gts), "nodes")


def test_node_at_dense_node_in_no_zone():
    g, zs, Psp = setup()
    nd = zs["nd"]
    none = np.flatnonzero(zs["zone_of"] < 0)
    assert len(none) >= 3
    P = nd["pos"][none] + [0.2, 0.0]                                                 # nearest to a dense node that is in no zone (a neighbour of an anchor), with a zone's node 2.0 m away
    c = counts(zs, P)
    assert c["n_off"] == len(none), "a node whose nearest dense GT node is in no zone is off-zone"
    # the toy has a no-zone node that has a zone node exactly 2 m away on both sides: 'any dense node within 2 m' would call it in-zone
    m = next(m for m in none if (np.linalg.norm(nd["pos"] - nd["pos"][m], axis=1) <= 2.0 + 1e-9).sum() > 1)
    assert counts(zs, [nd["pos"][m]])["n_off"] == 1
    print("test_node_at_dense_node_in_no_zone ok:", len(none), "such nodes")


def test_additive_pooling():
    g, zs, Psp = setup()
    g2, zs2, Psp2 = setup((0.3, 4.0))                                                # a second frame with other zone counts
    assert zs2["n_zones"] != zs["n_zones"]
    c1, c2 = counts(zs, Psp), counts(zs2, np.zeros((0, 2)), ln=60.0)
    tot = {k: c1[k] + c2[k] for k in ZM.COUNTS}
    tot["dens"] = [c1["dens"][0] + c2["dens"][0], c1["dens"][1] + c2["dens"][1]]
    a, b = ZM.pool([c1, c2]), ZM.pool([tot])
    for k in a:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k
    assert abs(a["free_zone_recall"] - c1["free_rec"] / (c1["n_free"] + c2["n_free"])) < 1e-12, "pooled = sum of counts, not the mean of the frame ratios"
    assert abs(a["nodes_per_100m"] - 100.0 * c1["n_deliv"] / (LEN + 60.0)) < 1e-12
    print("test_additive_pooling ok")


def test_bootstrap_diff():
    g, zs, Psp = setup()
    rng = np.random.RandomState(1)
    frames_of = lambda k, keep: [counts(zs, Psp[rng.rand(len(Psp)) < keep]) for _ in range(k)]     # every sparse GT node delivered with probability `keep`  # noqa: E731
    a = frames_of(30, 0.9)
    b = frames_of(30, 0.5)
    s = ZM.bootstrap_diff(a, a, "free_zone_recall")
    assert s["mean"] == 0.0 and s["lo90"] == 0.0 and s["hi90"] == 0.0 and s["median"] == 0.0 and s["point"] == 0.0, "a list against itself"
    d = ZM.bootstrap_diff(a, b, "free_zone_recall")
    assert d["lo90"] > 0 and d["lo90"] <= d["median"] <= d["hi90"] and d["lo90"] <= d["point"] <= d["hi90"] and 0.25 < d["mean"] < 0.55, d
    assert abs(d["point"] - (ZM.pool(a)["free_zone_recall"] - ZM.pool(b)["free_zone_recall"])) < 1e-12
    assert ZM.bootstrap_diff(a, b, "free_zone_recall") == d, "the same seed gives the same numbers"
    assert ZM.bootstrap_diff(a, b, "free_zone_recall", seed=1) != d
    r = ZM.bootstrap_diff(b, a, "free_zone_recall")
    assert r["hi90"] < 0 and abs(r["point"] + d["point"]) < 1e-12
    # identical frames: every resample gives the pooled difference of the full set
    same_a, same_b = [a[0]] * 10, [b[0]] * 10
    t = ZM.bootstrap_diff(same_a, same_b, "free_zone_exact")
    assert abs(t["lo90"] - t["hi90"]) < 1e-12 and abs(t["mean"] - t["point"]) < 1e-12
    # a metric that is NaN on some resamples (no delivered node in a frame set) does not break it
    e = [counts(zs, np.zeros((0, 2)))] * 5
    assert np.isnan(ZM.bootstrap_diff(e, e, "off_zone")["mean"])
    print("test_bootstrap_diff ok:", {k: round(v, 3) for k, v in d.items()})


def test_matched_threshold():
    fe = [np.array([0.9, 0.8, 0.45, 0.3, 0.1]), np.array([[0.95, 0.55], [0.2, 0.05]])]       # any shape; 9 values, the GT lane is 50 m long: density = 2 x (nodes with e >= thr)
    lens = [20.0, 30.0]
    r = ZM.matched_threshold(fe, lens, target=10.0)
    assert r["dens"] == 10.0 and r["thr"] == 0.31 and r["ok"], r
    r = ZM.matched_threshold(fe, lens, target=14.4)                                   # 7 nodes (14.0) is closer than 8 (16.0); the lowest threshold with 7 nodes
    assert r["dens"] == 14.0 and r["thr"] == 0.11 and r["ok"], r
    assert ZM.matched_threshold(fe, lens, target=12.0, grid=[0.3])["dens"] == 12.0, "e = 0.3 is delivered at thr 0.3 (e >= thr)"
    r = ZM.matched_threshold(fe, lens, target=100.0)
    assert not r["ok"] and r["dens"] == 18.0, "no threshold reaches the target: reported, not hidden"
    r = ZM.matched_threshold([np.array([0.3] * 6 + [0.9] * 2)], [50.0], target=4.0)       # the default grid holds 0.30000000000000004: the node with e = 0.3 is still delivered at 0.3, gone at 0.31
    assert r["thr"] == 0.31 and r["dens"] == 4.0, r
    print("test_matched_threshold ok")


def test_density_is_the_scorers():
    # the same delivered nodes, scored by qgraph_eval.node_counts and by frame_counts: the same `dens` (all delivered nodes over the GT lane length in the box)
    g, old = QE.toy_lane_gt()
    pred = qformat.oracle_prediction(g)
    ref = QE.LaneRef(g)
    zs = QZ.zones(g, SPARSE)
    P = np.asarray(pred["nodes"]["pos"], float)
    c = ZM.frame_counts(zs, ref.length, P)
    nc = QE.node_counts(pred, g)
    assert list(c["dens"]) == [float(x) for x in nc["dens"]] and c["dens"][1] == ref.length > 0
    assert abs(ZM.pool([c])["nodes_per_100m"] - QE.node_metrics(pred, g)["nodes_per_100m"]) < 1e-12
    print("test_density_is_the_scorers ok: nodes", int(c["dens"][0]), "lane length", ref.length)


if __name__ == "__main__":
    test_perfect_head()
    test_two_nodes_per_zone()
    test_empty_head()
    test_stray_node_and_radius()
    test_neighbour_cell_counts_by_position()
    test_node_at_dense_node_in_no_zone()
    test_additive_pooling()
    test_bootstrap_diff()
    test_matched_threshold()
    test_density_is_the_scorers()
    print("test_zone_metrics: all ok")

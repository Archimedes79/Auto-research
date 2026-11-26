"""Self-test of sparse_metrics: identical to scripts/p16_common.py, perfect graph scores 1, DP sparsity of the ground truth.
Run: scripts/_py14.sh ../evaluation/sparse_lanes/selftest.py"""
import os
import pickle

import numpy as np

import graph_eval as GE
import p16_common as OLD
import sparse_metrics as SM


def spec_graph_from_vertices():
    """spec of graph_from_vertices (review bug L1, D24): maximal chains between vertices with in- or out-degree != 1, every edge in exactly one piece"""
    def pieces(edges, n=8):
        xy = np.stack([np.arange(n, dtype=np.float64), np.zeros(n)], 1)
        g = SM.graph_from_vertices(xy, edges)
        assert sum(len(c) - 1 for c in g["nodes"]) == len({(a, b) for a, b in edges if a != b}), "every edge in exactly one piece"
        assert all(np.array_equal(p, xy[c]) for p, c in zip(g["pts"], g["nodes"]))
        return g["nodes"], sorted(g["edges"])
    cases = {  # name: (edges, pieces, piece edges)
        "isolated chain": ([(0, 1), (1, 2), (2, 3)], [[0, 1, 2, 3]], []),
        "fork": ([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)], [[0, 1, 2], [2, 3, 4], [2, 5, 6]], [(0, 1), (0, 2)]),
        "merge": ([(0, 1), (1, 2), (3, 4), (4, 2), (2, 5), (5, 6)], [[0, 1, 2], [2, 5, 6], [3, 4, 2]], [(0, 1), (2, 1)]),
        "fork at the first vertex": ([(0, 1), (1, 2), (0, 3), (3, 4)], [[0, 1, 2], [0, 3, 4]], []),
        "pure cycle": ([(1, 2), (2, 0), (0, 1)], [[0, 1, 2, 0]], []),
        "cycle with an entry": ([(3, 0), (0, 1), (1, 2), (2, 0)], [[0, 1, 2, 0], [3, 0]], [(1, 0)]),
        "two chains, self-loop and repeated edge dropped": ([(0, 1), (1, 2), (1, 2), (4, 4), (5, 6), (6, 7)], [[0, 1, 2], [5, 6, 7]], []),
    }
    for name, (edges, want_nodes, want_edges) in cases.items():
        got = pieces(edges)
        assert got == (want_nodes, want_edges), f"graph_from_vertices {name}: {got} != {(want_nodes, want_edges)}"
    # legacy_cut=True: the cut of the product path until route success is re-measured (D24) - the old stubs, exactly as before the fix
    xy = np.stack([np.arange(8, dtype=np.float64), np.zeros(8)], 1)
    old = SM.graph_from_vertices(xy, cases["fork"][0], legacy_cut=True)["nodes"]
    assert old == [[0, 1, 2], [3, 4], [5, 6], [2, 3], [2, 5]], f"legacy cut changed: {old}"
    print(f"graph_from_vertices spec: {len(cases)} cases passed (fork, merge, cycles, isolated chain) + the legacy cut")


def main():
    spec_graph_from_vertices()
    HOME = os.path.expanduser("~")
    gts = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
    print("gt graphs:", len(gts), "| keys of one graph:", sorted(gts[0].keys()) if isinstance(gts[0], dict) else type(gts[0]))
    gts = [dict(pts=[np.asarray(p, np.float64) for p in g["pts"]], edges=list(g["edges"]), nodes=g.get("nodes")) for g in gts[:60]]
    rng = np.random.RandomState(0)
    pred = [dict(pts=[p + rng.normal(0, 0.4, p.shape) for p in g["pts"]], edges=list(g["edges"]), nodes=None) for g in gts]
    for name, P in (("perfect", gts), ("noisy 0.4 m", pred)):
        a, b = OLD.card(P, gts), SM.card(P, gts)
        print(f"{name:12s} identical to p16_common: {a == b} | F1@1.0 {b['F1@1.0']:.3f}  J {b['J']:.3f}  Sp {b['Sp']:.3f}  S {b['S']:.3f}  vert100 {b['vert100']:.1f}")
        assert a == b
    print("vert100_dp on the ground-truth graphs (project oracle 14.3 on 100 TEST frames):", round(SM.vert100_dp(gts, 0.3), 2), "| unique end points:", round(SM.vert100_dp(gts, 0.3, unique_ends=True), 2))
    pv = [dict(pts=[np.array([[0.0, 0], [10, 0], [20, 0]])], edges=[], nodes=None)]
    print("straight 20 m line, dp vertices per 100 m:", round(SM.vert100_dp(pv, 0.3), 2), "(2 end points -> 10.0)")
    print("sparse_lanes selftest passed")


if __name__ == "__main__":
    main()

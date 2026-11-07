"""E276 tests (CPU): the tier metrics of the scorer (`qgraph_eval.tier_counts` / `tier_numbers`, in `score_frames`) on the toy lane with l4 labels, and the rule `verdict276` on fake cards.
Run: wsl bash tools/env/_py14.sh test_tier_metrics.py"""
import json
import os
import sys
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402
import qformat  # noqa: E402


def labelled_toy(vis_half_b=False):
    g, old = QE.toy_lane_gt()
    for p in g["pieces"]:
        n = len(p["pts"])
        p["tier"] = np.full(n, {"A": 1, "B": 2}.get(p["id"], 3), np.uint8)
        p["vis"] = np.ones(n, bool)
        if vis_half_b and p["id"] == "B":
            p["vis"][n // 2:] = False
    return g, old


def frame_of(g, old, k=0):
    class F(QE.Frame):
        def lane(self):
            return g

        def ref(self):
            if self._ref is None:
                self._ref = QE.LaneRef(g)
            return self._ref
    return F(k, k, f"tok{k}", k, old)


def raw_of(*polys):
    return dict(pts=[np.asarray(p, float) for p in polys], edges=[])


def numbers(raw, fr):
    c = QE.tier_counts(raw, fr.ref())
    return QE.tier_numbers(QE.pool_counts([c]))


_ROOTS = []                                                                 # temporary roots of the fake cards, removed at the end of main


def cards_root(spec):
    """spec: {name: dict(seed, t12, r1, topo, dens, exact, err)} -> a temporary root with runs/e276/<name>/card.json"""
    root = tempfile.mkdtemp()
    _ROOTS.append(root)
    for name, s in spec.items():
        d = os.path.join(root, "runs", "e276", name)
        os.makedirs(d)
        card = dict(seed=s["seed"], frames=873, p=dict(nodes=dict(nodes_per_100m=s["dens"]), graph=dict(topo_F1=s["topo"], vert100_dp=s["dens"], junction_F1=0.2), tier=dict(t12_F1_1=s["t12"], t1_R_1=s["r1"], t1_R_1all=s["r1"], t12_R_1=0.8, t12_P_1=0.8, z1_P_1=0.8, z2_P_1=0.8,
                                                                                                        z3_P_1=0.8)),
                    diff=dict(exact=dict(mean=s.get("exact", 0.0)), error=dict(mean=s.get("err", 0.05))))
        json.dump(card, open(os.path.join(d, "card.json"), "w"))
    return root


def main():
    g, old = labelled_toy()
    fr = frame_of(g, old)
    A, B, C = (next(p for p in g["pieces"] if p["id"] == i)["pts"] for i in "ABC")
    # 1. the GT itself delivered: everything right, in every tier and zone
    n = numbers(raw_of(A, B, C), fr)
    assert n["t1_R_1"] == 1.0 and n["t2_R_1"] == 1.0 and n["t3_R_1"] == 1.0 and n["z1_P_1"] == 1.0 and n["z2_P_1"] == 1.0 and n["t12_F1_1"] == 1.0, n
    assert abs(n["t1_len"] - 0.5 * len(A)) < 1e-9, "t1_len is the visible GT length of tier 1 in metres"
    # 2. a null graph scores 0 (not 1), on the primary number too
    n0 = numbers(raw_of(), fr)
    assert n0["t1_R_1"] == 0.0 and n0["t12_F1_1"] == 0.0 and n0["z1_P_1"] == 0.0, n0
    # 3. without the tier-1 lane: tier-1 recall 0, tier 2 and 3 untouched
    n3 = numbers(raw_of(B, C), fr)
    assert n3["t1_R_1"] < 0.1 and n3["t2_R_1"] == 1.0 and n3["t3_R_1"] == 1.0, n3                 # the last metres of A touch the start of B (within 1 m), the rest of A is not delivered
    # 4. a copy of the own lane shifted by 1.6 m: half of the points in zone 1 are wrong, tier-1 recall stays 1 (the original is there)
    n4 = numbers(raw_of(A, B, C, A + np.array([1.6, 0.0])), fr)
    assert n4["t1_R_1"] == 1.0 and abs(n4["z1_P_1"] - 0.5) < 0.05 and n4["t12_P_1"] < n["t12_P_1"], n4
    # 5. visibility: an invisible GT part is not demanded (the visible recall ignores it) but the `all` recall counts it
    gv, oldv = labelled_toy(vis_half_b=True)
    frv = frame_of(gv, oldv)
    Bv = next(p for p in gv["pieces"] if p["id"] == "B")["pts"]
    n5 = numbers(raw_of(A, C, Bv[: len(Bv) // 2]), frv)
    assert n5["t2_R_1"] == 1.0 and abs(n5["t2_R_1all"] - 0.5) < 0.05 and abs(n5["t2_len"] - 0.25 * (len(Bv) - 1)) < 1.0, n5
    # 6. a graph without l4 labels: no tier counts, no tier numbers
    gu, oldu = QE.toy_lane_gt()
    fru = frame_of(gu, oldu)
    assert QE.tier_counts(raw_of(A, B, C), fru.ref()) is None and not fru.ref().labelled
    # 7. the pooled numbers are the sums (two frames, one of them without the tier-1 lane)
    cs = [QE.tier_counts(raw_of(A, B, C), fr.ref()), QE.tier_counts(raw_of(B, C), fr.ref())]
    p = QE.tier_numbers(QE.pool_counts(cs))
    assert abs(p["t1_R_1"] - 0.5 * (1.0 + numbers(raw_of(B, C), fr)["t1_R_1"])) < 1e-9, p        # (1 + the junction contact of frame two) / 2: the counts are summed, not the ratios averaged
    # 8. through score_frames with the oracle: tier numbers and node-level tier numbers are there; a graph without labels gives {}
    pred = qformat.oracle_prediction(g)
    out = QE.score_frames([pred], [fr])
    assert out["tier"]["t12_F1_1"] > 0.95 and out["tier"]["t1_node_R"] > 0.9 and out["tier"]["z1_node_P"] > 0.9, out["tier"]
    assert QE.score_frames([qformat.oracle_prediction(gu)], [fru])["tier"] == {}
    tr = out["tier"]
    assert all(f"z{k}_node_n" in tr and f"z{k}_node_false" in tr for k in range(4)) and tr["z1_node_n"] > 0, "delivered nodes and false nodes per zone"
    evk = [k for k in tr if k.startswith("t1_ev_") and k.endswith("_n") and tr[k] > 0]
    assert evk and all(tr[k[:-2] + "_R"] >= 0.5 for k in evk), (evk, {k: tr[k[:-2] + "_R"] for k in evk})            # the oracle finds the tier-1 event nodes
    # a raw graph with only two vertices 40 m apart is resampled before it is counted (the reference d has sparse vertices): the recall is not understated
    assert numbers(raw_of(np.array([A[0], A[-1]])), fr)["t1_R_1"] > 0.95
    # 9. the rule verdict276 on fake cards
    def spec(t12=(0.80, 0.80), r1=(0.9, 0.9), topo=(0.60, 0.60), dens=(30.0, 30.0), exact=(0.0, 0.0), err=(0.05, 0.05), pre="v"):
        return {f"{pre}_s{k}": dict(seed=k, t12=t12[k], r1=r1[k], topo=topo[k], dens=dens[k], exact=exact[k], err=err[k]) for k in (0, 1)}
    ctl = spec(pre="c")
    for kind, var_kw, want in [("t", dict(t12=(0.84, 0.85)), "KEEP"), ("t", dict(t12=(0.81, 0.82)), "TIE"), ("t", dict(t12=(0.84, 0.77)), "CUT"), ("t", dict(t12=(0.84, 0.81)), "INCONCLUSIVE"),
                               ("t", dict(t12=(0.84, 0.85), topo=(0.54, 0.60)), "CUT"), ("t", dict(t12=(0.84, 0.85), r1=(0.87, 0.9)), "CUT"),
                               ("t", dict(t12=(0.84, 0.85), exact=(-0.02, -0.02)), "CUT"), ("t", dict(t12=(0.84, 0.85), err=(0.02, 0.02)), "CUT"),
                               ("s", dict(dens=(20.0, 21.0)), "KEEP"), ("s", dict(dens=(28.0, 29.0)), "TIE"), ("s", dict(dens=(20.0, 21.0), t12=(0.77, 0.80)), "CUT")]:
        root = cards_root({**ctl, **spec(pre="v", **var_kw)})
        v = QR.verdict276(["v_s0", "v_s1"], ["c_s0", "c_s1"], "e276", "e276", kind, root=root, verbose=False)
        assert v["verdict"].startswith(want), (kind, var_kw, v["verdict"], v["guards"], v["t12_F1_1_gain_per_seed"])
    # the checks: wrong seeds, a card without tier numbers
    root = cards_root({**ctl, **spec(pre="v")})
    try:
        QR.verdict276(["v_s1", "v_s0"], ["c_s0", "c_s1"], "e276", "e276", "t", root=root, verbose=False)
        raise SystemExit("the seed order was not checked")
    except AssertionError:
        pass
    for r in _ROOTS:
        shutil.rmtree(r, ignore_errors=True)
    print("test_tier_metrics: tier recall / precision / F1 on the toy lane (oracle, null, missing own lane, shifted copy, invisible part, no labels, pooling), score_frames, verdict276 (t and s, every guard): ok")


if __name__ == "__main__":
    main()

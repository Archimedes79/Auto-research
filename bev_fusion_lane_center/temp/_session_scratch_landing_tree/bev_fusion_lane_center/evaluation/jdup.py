"""Where do the duplicate points of dup_frac come from? (E324 diagnosis, promoted from the scratch scripts of 2026-11-13)

    wsl bash tools/env/_py14.sh jdup.py score --seed 0 [--a-run e320_a_s0] [--c-run e312_p_s0] [--max N]     -> temp/jdup/s<seed>.json + a printed table (CPU, VAL, about one minute)
    wsl bash tools/env/_py14.sh jdup.py selftest

Same counting as `sparse_lanes/topo_metrics.dup_counts` (dense points every 0.5 m; a point is a duplicate when another piece that is neither its graph neighbour nor its sibling runs parallel (|cos| > 0.9) within 0.75 m; the first and last
4 m of a piece never count), but every duplicate point is attributed, for six read-outs (all on B's delivered graph with the hard step `close:1.5@0.5`, connectors at p >= 0.1, snapped to the delivered pieces):
  c  E312 snapped (keypoint ends, the snap's geometry)      u  E312 snap-nogeom (same curves as c, keypoint geometry)      k  E320 own keypoints snapped      kn  E320 own keypoints snap-nogeom
  a  E320 PRIMARY (attached ends, snap-nogeom)              l  a third run read out like c (--l-run, e.g. E316)
  by kind        own piece / partner piece = connector (C) or B piece (B): CC (no B partner), CB (a B partner), BC, BB, and `conn_dup_with_c_partner` = the connector duplicate points that have a connector partner (CC + `CBc`, the CB points that also have one); for CB the arc length from the nearest end of the
                 connector (4 - 8, 8 - 16, 16+ m) and the kind of the B piece (section / link)
  by truth       a connector is TRUE when both of its ends match the two ends of one GT pair (strict one-to-one matching of `jwiring.match`, 4 m, 30 degrees), FALSE otherwise (wrong pairs, unmatched or duplicate keypoints, unannotated lanes
                 are not told apart): curves, curve points and duplicate points of each
  by probability the pair probability bins 0.1 - 0.3 / 0.3 - 0.5 / 0.5 - 0.8 / 0.8+ : curves, share true, duplicate share of the curve points
The totals reproduce the dup_frac of the cards (c 0.1747 / 0.1557, a 0.1931 / 0.1651 on VAL). Needs the GT ends of the VAL frames (`jwiring.py gt-ends`).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "evaluation/sparse_lanes", "training", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import graph_eval as GE  # noqa: E402
import topo_metrics as TM  # noqa: E402

ARC = ((4.0, 8.0, "arc_4_8"), (8.0, 16.0, "arc_8_16"), (16.0, 1e9, "arc_16p"))


def _skip_pairs(pred):
    """(i, j) pairs that never count: graph neighbours and siblings (branches of one split / merge), as in topo_metrics.dup_counts"""
    near = set()
    for i, j in pred["edges"]:
        near.add((i, j))
        near.add((j, i))
    succ, pre = {}, {}
    for i, j in pred["edges"]:
        succ.setdefault(i, set()).add(j)
        pre.setdefault(j, set()).add(i)
    for grp in list(succ.values()) + list(pre.values()):
        for a in grp:
            for b in grp:
                if a != b:
                    near.add((a, b))
    return near


def attribute(pred, n_conn, kinds=None, r=0.75, cos_min=0.9, end_zone=4.0):
    """pred = the graph dict of the scorer (pts, edges, nodes); the LAST n_conn pieces are the connectors, in the order of the pairs; kinds = the kinds ('section' / 'link') of the B pieces, same order, optional.
    -> (total, per_conn) with total = dict(points, dup, CC, CB, BC, BB, conn_points, arc_*, partner_section, partner_link) and per_conn = (dup points per connector, points per connector)"""
    P, D, I = GE.dense_points(pred)
    t = dict(points=len(P), dup=0, CC=0, CB=0, BC=0, BB=0, conn_points=0, arc_4_8=0, arc_8_16=0, arc_16p=0, partner_section=0, partner_link=0, conn_dup_with_c_partner=0, CBc=0)
    dups, pts = np.zeros(n_conn, int), np.zeros(n_conn, int)
    if len(P) < 2:
        return t, (dups, pts)
    nB = len(pred["pts"]) - n_conn
    isC = I >= nB
    t["conn_points"] = int(isC.sum())
    for i in I[isC]:
        pts[int(i) - nB] += 1
    margin = TM._arc_margin(pred)
    near = _skip_pairs(pred)
    tree = cKDTree(P)
    for k, nb in enumerate(tree.query_ball_point(P, r)):
        if margin[k] < end_zone:
            continue
        partners = [j for j in nb if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > cos_min]
        if partners:
            has_c = any(isC[j] for j in partners)
            has_b = any(not isC[j] for j in partners)
            j = next((j for j in partners if not isC[j]), partners[0])                                      # the B partner when there is one (the indices of B are the lower ones, as in dup_counts' first hit)
            t["dup"] += 1
            t[("C" if isC[k] else "B") + ("C" if isC[j] else "B")] += 1
            if isC[k]:
                dups[int(I[k]) - nB] += 1
                t["conn_dup_with_c_partner"] += int(has_c)                                                 # ALL connector duplicate points that have a connector partner: the CC points (no B partner) and the CB points that also have one
                if has_b:
                    t["CBc"] += int(has_c)                                                                 # of the CB points: those that ALSO have a connector partner (hidden behind the B partner in the CB class)
                    for lo, hi, name in ARC:
                        if lo <= margin[k] < hi:
                            t[name] += 1
                    if kinds is not None:
                        t["partner_link" if kinds[int(I[j])] == "link" else "partner_section"] += 1
    return t, (dups, pts)


# the read-outs that are attributed: name, which run's dump, attached ends, snap-nogeom wiring (all snapped, connectors at p >= 0.1, B's delivered graph with the hard step)
VARIANTS = (("c", "c_run", False, False), ("u", "c_run", False, True), ("k", "a_run", False, False), ("kn", "a_run", False, True), ("a", "a_run", True, True), ("l", "l_run", False, False))
NAMES = dict(c="c  = E312 snapped (keypoint ends, snap geometry)", u="u' = E312 snap-nogeom (the floor, same curves as c)", k="k  = E320 own keypoints, snapped", kn="kn = E320 own keypoints, snap-nogeom",
             a="a  = E320 PRIMARY (attached ends, snap-nogeom)", l="l  = the third run, snapped (--l-run, e.g. E316)")


def _load(a):
    import qgraph_eval as QE
    import qgraph_readout as QR
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
    runs = dict(c_run=a.c_run or f"e312_p_s{a.seed}", a_run=a.a_run or f"e320_a_s{a.seed}", l_run=a.l_run)
    ends = {r: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in set(v for v in runs.values() if v)}
    return QE, QR, dump, ends, runs


def score(a):
    import jwiring as JW
    from jwire import kept_ends
    from qends_eval import cover_pos, add_connectors, make_pairs
    QE, QR, dump, ends, runs = _load(a)
    gt = pickle.load(open(JW.GT_PATH, "rb"))
    frames, spec = QE.val_frames(), QR.parse("close:1.5@0.5")
    use = [(n, runs[r], att, ng) for n, r, att, ng in VARIANTS if runs.get(r)]
    keys = ("points", "dup", "CC", "CB", "BC", "BB", "conn_points", "arc_4_8", "arc_8_16", "arc_16p", "partner_section", "partner_link", "conn_dup_with_c_partner", "CBc")
    bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    tot = {n: dict({k: 0 for k in keys}, run=run, curves=0, truth={tr: dict(curves=0, points=0, dup=0) for tr in ("true", "false")}, bins={str(b): dict(curves=0, points=0, dup=0, true=0) for b in bins}) for n, run, _, _ in use}
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if a.max and b >= a.max:
            break
        pr = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                           links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QE.readout(pr, 0.5, 0.5)
        lm = np.asarray(pr["links"]["long"], bool)
        g = gt[f.row]
        gtp = {(int(x), int(y)) for x, y in g["pairs"] if x >= 0}
        for v, run, att, ng in use:
            e = ends[run][f.row]
            k = e["kp_pos"].shape[0] // 2
            pm = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
            ia, ib = np.nonzero(pm >= 0.1)
            valid = np.asarray(e["valid"], bool)
            m_in = JW.match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
            m_out = JW.match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
            pairs = make_pairs(e, ia, ib, k, attach=att)
            raw, n_c, _ = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], lm, pairs, snap=True, snap_nogeom=ng)
            kept = [bk["kind"][i] for i in kept_ends(graph.lines, bk["kind"], bk["piece_links"], lm, pairs, cover_pos())[0]]                       # the kinds of the pieces the cover drop keeps (E325: one place for it)
            assert len(kept) == len(raw["pts"]) - n_c, (len(kept), len(raw["pts"]), n_c)
            t, (dups, pts) = attribute(raw, n_c, kept)
            T = tot[v]
            for kk in keys:
                T[kk] += t[kk]
            T["curves"] += n_c
            for n_, (i, j) in enumerate(zip(ia, ib)):
                tr = "true" if m_in[i] >= 0 and m_out[j] >= 0 and (int(m_in[i]), int(m_out[j])) in gtp else "false"
                T["truth"][tr]["curves"] += 1
                T["truth"][tr]["points"] += int(pts[n_])
                T["truth"][tr]["dup"] += int(dups[n_])
                for bn in bins:
                    if bn[0] <= float(pm[i, j]) < bn[1]:
                        bb = T["bins"][str(bn)]
                        bb["curves"] += 1
                        bb["points"] += int(pts[n_])
                        bb["dup"] += int(dups[n_])
                        bb["true"] += int(tr == "true")
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{len(frames)} {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.join(ROOT, "temp", "jdup"), exist_ok=True)
    json.dump(tot, open(os.path.join(ROOT, "temp", "jdup", f"s{a.seed}{'_max' + str(a.max) if a.max else ''}.json"), "w"), indent=1)
    for v, run, _, _ in use:
        T = tot[v]
        print(f"{NAMES[v]} [{run}]: points {T['points']} dup {T['dup']} ({T['dup'] / max(T['points'], 1):.4f}); CC (no B partner) {T['CC']} CB {T['CB']} BC {T['BC']} BB {T['BB']}; connector duplicate points with a connector partner {T['conn_dup_with_c_partner']} = CC + CB-with-a-connector-partner {T['CBc']} ({100 * T['conn_dup_with_c_partner'] / max(T['CC'] + T['CB'], 1):.1f} % of the connector duplicate points CC + CB); "
              f"curves {T['curves']} (points {T['conn_points']}); CB by arc 4-8 / 8-16 / 16+: {T['arc_4_8']} / {T['arc_8_16']} / {T['arc_16p']}, partner section / link {T['partner_section']} / {T['partner_link']}")
        print(f"   curves true / false {T['truth']['true']['curves']} / {T['truth']['false']['curves']}; duplicate points of connectors true / false {T['truth']['true']['dup']} / {T['truth']['false']['dup']}; per curve {T['truth']['true']['dup'] / max(T['truth']['true']['curves'], 1):.1f} / {T['truth']['false']['dup'] / max(T['truth']['false']['curves'], 1):.1f}")
        for bn, bb in T["bins"].items():
            print(f"   p in {bn}: {bb['curves']} curves ({100 * bb['true'] / max(bb['curves'], 1):.0f} % true), duplicate share {100 * bb['dup'] / max(bb['points'], 1):.1f} %")


def selftest():
    # three B pieces (a long straight lane, a lane 6 m beside it, a short one) and two connectors: one runs 0.4 m beside the first lane (not its neighbour), one is far from everything
    lane = lambda x0, y0, y1: np.c_[np.zeros(200) + x0, np.linspace(y0, y1, 200)]
    pred = dict(pts=[lane(0.0, -30, 30), lane(6.0, -30, 30), lane(-12.0, -5, 5), lane(0.4, -15, 15), lane(20.0, -30, 30)], edges=[], nodes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    t, (dups, pts) = attribute(pred, 2, ["section", "section", "section"])
    ref = TM.dup_counts(pred)
    assert t["dup"] == ref[0] and t["points"] == ref[1], (t, ref)                                      # the attribution adds up to the metric
    assert t["BC"] > 0 and t["CB"] > 0 and t["CC"] == 0 and t["BB"] == 0 and t["partner_section"] == t["CB"], t
    assert dups[0] > 0 and dups[1] == 0 and pts[0] > 0 and pts[1] > 0, (dups, pts)                     # the connector beside the lane has duplicate points, the far one none
    assert t["arc_4_8"] + t["arc_8_16"] + t["arc_16p"] == t["CB"], t
    # a graph neighbour is never a duplicate: the same connector linked to the lane
    pred2 = dict(pred, edges=[(0, 3)])
    t2, _ = attribute(pred2, 2, ["section", "section", "section"])
    assert t2["CB"] == 0 and t2["dup"] == TM.dup_counts(pred2)[0], t2
    # two connectors that run side by side: connector-connector duplicates
    pred3 = dict(pts=[lane(0.0, -30, 30), lane(20.0, -30, 30), lane(10.0, -20, 20), lane(10.4, -20, 20)], edges=[], nodes=[[0, 1], [2, 3], [4, 5], [6, 7]])
    t3, _ = attribute(pred3, 2, ["section", "section"])
    assert t3["CC"] > 0 and t3["CB"] == 0 and t3["dup"] == TM.dup_counts(pred3)[0], t3
    # the connector-partner key: CC points count once (key == CC, CBc == 0), a lane with no connector near it has key 0, a connector pair beside a lane has both a B and a connector partner (CBc == key - CC)
    assert t3["conn_dup_with_c_partner"] == t3["CC"] and t3["CBc"] == 0, t3
    assert t["conn_dup_with_c_partner"] == 0 and t["CBc"] == 0, t
    pred4 = dict(pts=[lane(0.0, -30, 30), lane(30.0, -30, 30), lane(0.3, -20, 20), lane(-0.3, -20, 20)], edges=[], nodes=[[0, 1], [2, 3], [4, 5], [6, 7]])
    t4, _ = attribute(pred4, 2, ["section", "section"])
    assert t4["CB"] > 0 and t4["CBc"] == t4["CB"] and t4["conn_dup_with_c_partner"] == t4["CC"] + t4["CBc"] and t4["dup"] == TM.dup_counts(pred4)[0], t4
    print("jdup selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--c-run", default=None)
    s.add_argument("--a-run", default=None)
    s.add_argument("--l-run", default=None, help="a third run (e.g. e316_p_s0), read out snapped like c")
    s.add_argument("--max", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        score(a)

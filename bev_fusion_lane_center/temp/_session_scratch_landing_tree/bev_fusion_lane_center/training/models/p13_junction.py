"""Loop 13 / E113b: where do the junctions get lost? Diagnosis of the split / merge locations of a traced graph against the ground-truth graph (TEST, all rows).
For every ground-truth split (end of a piece with >= 2 successors) / merge (start of a piece with >= 2 predecessors): detected (a predicted junction of the same kind within 6 m), else: 'children covered' (all branch pieces of the junction are covered by predicted points within 1.5 m: the topology is missing, a link or a junction event would repair it),
'a branch missing' (some branch piece is covered by less than 60 %: a lane is missing), 'lane missed' (no predicted point within 3 m of the junction).
For every predicted junction: true (a ground-truth junction of the same kind within 6 m), else: 'on a real lane' (ground-truth centre line within 1.5 m) or 'false lane'.
Systems: learned trace network (E94a, seed 0), the rule pipeline's loose graph, its final graph (key points + connectors).
Usage (gcar, CPU): p13_junction.py --out e113_a"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import p8_lib as L  # noqa: E402

OUT = L.OUT


def kinds(g):
    """-> list of (kind, location, piece ids involved) for the junctions of a graph"""
    n = len(g["pts"])
    out_deg, in_deg = np.zeros(n, int), np.zeros(n, int)
    succ, pred = {}, {}
    for i, j in g["edges"]:
        out_deg[i] += 1
        in_deg[j] += 1
        succ.setdefault(i, []).append(j)
        pred.setdefault(j, []).append(i)
    res = []
    for i in range(n):
        if len(g["pts"][i]) == 0:
            continue
        if out_deg[i] >= 2:
            res.append(("split", np.asarray(g["pts"][i][-1], float), [i] + succ[i]))
        if in_deg[i] >= 2:
            res.append(("merge", np.asarray(g["pts"][i][0], float), [i] + pred[i]))
    return res


def points(g, ds=1.0):
    xy, _, _, _ = GE.point_graph(g, ds)
    return xy


def cover(piece_pts, tree, r=1.5, ds=1.0):
    q = GE.resample(np.asarray(piece_pts, float), ds)
    if len(q) == 0:
        return 0.0
    d, _ = tree.query(q)
    return float((d <= r).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e113_a")
    a = ap.parse_args()
    src = L.load_src("test_clean")
    ids = [int(i) for i in src["ids"]]
    gts = GD.load_gt_graphs("val", ids)
    systems = {
        "learned (E94a)": pickle.load(open(f"{OUT}/e94_a/graphs_test_clean.pkl", "rb"))["preds"],
        "rule, loose graph": src["graphs"],
        "rule, key points + connectors": pickle.load(open(f"{OUT}/e53_f0.50/graphs_test.pkl", "rb"))["graphs"],
    }
    res = {}
    for name, preds in systems.items():
        cnt = {k: dict(n=0, detected=0, children_covered=0, branch_missing=0, lane_missed=0, pred_n=0, pred_true=0, pred_real_lane=0, pred_false_lane=0) for k in ("split", "merge")}
        for g, p in zip(gts, preds):
            gj, pj = kinds(g), kinds(p)
            ptree = cKDTree(points(p)) if len(points(p)) else None
            gtree = cKDTree(points(g)) if len(points(g)) else None
            for kind in ("split", "merge"):
                G_ = [x for x in gj if x[0] == kind]
                P_ = [x for x in pj if x[0] == kind]
                pl = np.array([x[1] for x in P_]).reshape(-1, 2)
                gl = np.array([x[1] for x in G_]).reshape(-1, 2)
                for _, loc, pieces in G_:
                    c = cnt[kind]
                    c["n"] += 1
                    if len(pl) and np.linalg.norm(pl - loc, axis=1).min() <= 6.0:
                        c["detected"] += 1
                        continue
                    if ptree is None or ptree.query(loc)[0] > 3.0:
                        c["lane_missed"] += 1
                        continue
                    cov = [cover(g["pts"][k], ptree) for k in pieces[1:]]  # the branch pieces
                    if min(cov) >= 0.6:
                        c["children_covered"] += 1
                    else:
                        c["branch_missing"] += 1
                for _, loc, pieces in P_:
                    c = cnt[kind]
                    c["pred_n"] += 1
                    if len(gl) and np.linalg.norm(gl - loc, axis=1).min() <= 6.0:
                        c["pred_true"] += 1
                    elif gtree is not None and gtree.query(loc)[0] <= 1.5:
                        c["pred_real_lane"] += 1
                    else:
                        c["pred_false_lane"] += 1
        res[name] = cnt
        for kind in ("split", "merge"):
            c = cnt[kind]
            n, pn = max(c["n"], 1), max(c["pred_n"], 1)
            print(f"{name:32s} {kind:5s} GT junctions {c['n']:4d}: detected {c['detected'] / n:.2f}, branches covered but no junction {c['children_covered'] / n:.2f}, a branch missing {c['branch_missing'] / n:.2f}, lane missed {c['lane_missed'] / n:.2f} | predicted {c['pred_n']:4d}: true {c['pred_true'] / pn:.2f}, on a real lane {c['pred_real_lane'] / pn:.2f}, false lane {c['pred_false_lane'] / pn:.2f}", flush=True)
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    json.dump(res, open(f"{OUT}/{a.out}/junction_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()

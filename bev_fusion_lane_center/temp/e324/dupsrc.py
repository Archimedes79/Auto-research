"""E324 scratch: WHERE do the duplicate points of dup_frac come from? Same counting as topo_metrics.dup_counts (dense points every 0.5 m, parallel within 0.75 m, |cos| > 0.9, neither neighbour nor sibling, not in the first / last 4 m),
but every duplicate point is attributed to (the type of its own piece, the type of the piece it is parallel to): connector (C) or B piece (B). For the snapped read-out of E312 (variant c) and the E320 PRIMARY read-out of a seed. CPU, one thread.
usage: dupsrc.py <seed> [--max N]   -> temp/e324/dupsrc_s<seed>.json"""
import json, os, pickle, sys, time
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "evaluation/sparse_lanes", "training", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
import qgraph_readout as QR
import graph_eval as GE
import topo_metrics as TM
from qends_eval import add_connectors, make_pairs
S = int(sys.argv[1]); MAXF = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{S}.pkl"), "rb"))
frames = QE.val_frames(); spec = QR.parse("close:1.5@0.5")
E = {n: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for n, r in (("c", f"e312_p_s{S}"), ("a", f"e320_a_s{S}"))}


def attribute(pred, n_conn, r=0.75, cos_min=0.9, end_zone=4.0):
    """-> dict(points, dup, and the duplicate points by (own type, partner type)) of one frame; the first len(pts) - n_conn pieces are B's, the last n_conn the connectors"""
    P, D, I = GE.dense_points(pred)
    out = dict(points=len(P), dup=0, CC=0, CB=0, BC=0, BB=0, conn_points=0, conn_with_dup=0)
    if len(P) < 2:
        return out
    nB = len(pred["pts"]) - n_conn
    isC = I >= nB
    out["conn_points"] = int(isC.sum())
    margin = TM._arc_margin(pred)
    near = set()
    for i, j in pred["edges"]:
        near.add((i, j)); near.add((j, i))
    succ, pre = {}, {}
    for i, j in pred["edges"]:
        succ.setdefault(i, set()).add(j); pre.setdefault(j, set()).add(i)
    for grp in list(succ.values()) + list(pre.values()):
        for a in grp:
            for b in grp:
                if a != b:
                    near.add((a, b))
    tree = cKDTree(P)
    seen_c = set()
    for k, nb in enumerate(tree.query_ball_point(P, r)):
        if margin[k] < end_zone:
            continue
        for j in nb:
            if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > cos_min:
                out["dup"] += 1
                out[("C" if isC[k] else "B") + ("C" if isC[j] else "B")] += 1
                if isC[k]:
                    seen_c.add(int(I[k]))
                break
    out["conn_with_dup"] = len(seen_c)
    return out


tot = {v: dict(points=0, dup=0, CC=0, CB=0, BC=0, BB=0, conn_points=0, conn_with_dup=0, curves=0) for v in ("c", "a")}
t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    if MAXF and b >= MAXF:
        break
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    for v in ("c", "a"):
        e = E[v][f.row]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        pairs = make_pairs(e, ia, ib, k, attach=(v == "a"))
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, snap=True, snap_nogeom=(v == "a"))
        r_ = attribute(raw, n_c)
        for kk in r_:
            tot[v][kk] += r_[kk]
        tot[v]["curves"] += n_c
    f._lane = f._ref = None
    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{len(frames)} {time.time() - t0:.0f} s", flush=True)
json.dump(tot, open(os.path.join(ROOT, "temp", "e324", f"dupsrc_s{S}{'_max' + str(MAXF) if MAXF else ''}.json"), "w"), indent=1)
for v, name in (("c", "E312 snapped (c)"), ("a", "E320 PRIMARY (a)")):
    t = tot[v]
    print(f"{name}: points {t['points']} dup {t['dup']} ({t['dup'] / t['points']:.4f}); duplicate points by (own, partner): CC {t['CC']} CB {t['CB']} BC {t['BC']} BB {t['BB']}; connector points {t['conn_points']}, curves {t['curves']}, connectors with a duplicate point {t['conn_with_dup']}")

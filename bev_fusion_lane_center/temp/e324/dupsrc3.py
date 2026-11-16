"""E324 scratch, step 3: the duplicate points of the connectors split by the TRUTH of the connector (true pair: both ends match the two ends of one GT pair; false pair otherwise) and by the pair probability, for E312 (c) and E320 PRIMARY (a). usage: dupsrc3.py <seed>"""
import json, os, pickle, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "evaluation/sparse_lanes", "training", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
import qgraph_readout as QR
import graph_eval as GE
import topo_metrics as TM
import jwiring as JW
from qends_eval import add_connectors, make_pairs
S = int(sys.argv[1])
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{S}.pkl"), "rb"))
gt = pickle.load(open(JW.GT_PATH, "rb"))
frames = QE.val_frames(); spec = QR.parse("close:1.5@0.5")
E = {n: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for n, r in (("c", f"e312_p_s{S}"), ("a", f"e320_a_s{S}"))}
BINS = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
tot = {v: {"n_curves": {"true": 0, "false": 0}, "curve_points": {"true": 0, "false": 0}, "dup_points": {"true": 0, "false": 0}, "bins": {str(b): dict(n=0, pts=0, dup=0, true=0) for b in BINS}} for v in ("c", "a")}


def conn_dups(pred, nB, n_conn):
    """duplicate points per connector index (0..n_conn-1) and the points per connector"""
    P, D, I = GE.dense_points(pred)
    dups = np.zeros(n_conn, int); pts = np.zeros(n_conn, int)
    if len(P) < 2:
        return dups, pts
    for i in I[I >= nB]:
        pts[int(i) - nB] += 1
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
    for k, nb in enumerate(tree.query_ball_point(P, 0.75)):
        if margin[k] < 4.0 or I[k] < nB:
            continue
        for j in nb:
            if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > 0.9:
                dups[int(I[k]) - nB] += 1
                break
    return dups, pts


for b, (f, d) in enumerate(zip(frames, dump)):
    pr = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                       links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pr, 0.5, 0.5)
    g = gt[f.row]
    gtp = {(int(a), int(c)) for a, c in g["pairs"] if a >= 0}
    for v in ("c", "a"):
        e = E[v][f.row]
        k = e["kp_pos"].shape[0] // 2
        pm = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(pm >= 0.1)
        valid = np.asarray(e["valid"], bool)
        m_in = JW.match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
        m_out = JW.match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
        pairs = make_pairs(e, ia, ib, k, attach=(v == "a"))
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pr["links"]["long"], bool), pairs, snap=True, snap_nogeom=(v == "a"))
        nB = len(raw["pts"]) - n_c
        dups, pts = conn_dups(raw, nB, n_c)
        for n_, (i, j) in enumerate(zip(ia, ib)):
            tr = "true" if (int(m_in[i]), int(m_out[j])) in gtp and m_in[i] >= 0 and m_out[j] >= 0 else "false"
            t = tot[v]
            t["n_curves"][tr] += 1; t["curve_points"][tr] += int(pts[n_]); t["dup_points"][tr] += int(dups[n_])
            p_ = float(pm[i, j])
            for bn in BINS:
                if bn[0] <= p_ < bn[1]:
                    bb = t["bins"][str(bn)]; bb["n"] += 1; bb["pts"] += int(pts[n_]); bb["dup"] += int(dups[n_]); bb["true"] += int(tr == "true")
    f._lane = f._ref = None
json.dump(tot, open(os.path.join(ROOT, "temp", "e324", f"dupsrc3_s{S}.json"), "w"), indent=1)
for v in ("c", "a"):
    t = tot[v]
    print(v, "curves", t["n_curves"], "points", t["curve_points"], "dup points", t["dup_points"])
    for bn, bb in t["bins"].items():
        print(f"   p in {bn}: {bb['n']} curves ({100 * bb['true'] / max(bb['n'], 1):.0f} % true), dup points {bb['dup']} of {bb['pts']} ({100 * bb['dup'] / max(bb['pts'], 1):.1f} %)")

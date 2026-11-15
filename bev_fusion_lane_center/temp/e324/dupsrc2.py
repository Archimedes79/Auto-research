"""E324 scratch, step 2: for the duplicate points of CONNECTORS (own = connector) that are parallel to a B piece (CB): (a) the arc length from the nearest end of the connector (4 - 8 m / 8 - 16 m / > 16 m), (b) the kind of the B piece (section / link),
(c) the graph hops between the connector and that piece (2 = a neighbour of a neighbour, 3+ = further). Same counting as dupsrc.py. usage: dupsrc2.py <seed> [--max N]"""
import json, os, pickle, sys, time
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "evaluation/sparse_lanes", "training", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
import qgraph_readout as QR
import graph_eval as GE
import topo_metrics as TM
from qends_eval import add_connectors, make_pairs, COVER_POS
S = int(sys.argv[1]); MAXF = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{S}.pkl"), "rb"))
frames = QE.val_frames(); spec = QR.parse("close:1.5@0.5")
E = {n: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for n, r in (("c", f"e312_p_s{S}"), ("a", f"e320_a_s{S}"))}
KEYS = ["arc_4_8", "arc_8_16", "arc_16p", "partner_section", "partner_link", "hops2", "hops3p", "cb_total"]
tot = {v: {k: 0 for k in KEYS} for v in ("c", "a")}


def hops(edges, n, src):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    d = {src: 0}; q = deque([src])
    while q:
        u = q.popleft()
        if d[u] >= 4:
            continue
        for w in adj[u]:
            if w not in d:
                d[w] = d[u] + 1; q.append(w)
    return d


def run(raw, n_conn, kinds_kept, res):
    pred = raw
    P, D, I = GE.dense_points(pred)
    if len(P) < 2:
        return
    nB = len(pred["pts"]) - n_conn
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
    hcache = {}
    for k, nb in enumerate(tree.query_ball_point(P, 0.75)):
        if margin[k] < 4.0 or I[k] < nB:
            continue
        for j in nb:
            if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > 0.9:
                if I[j] < nB:
                    res["cb_total"] += 1
                    m = margin[k]
                    res["arc_4_8" if m < 8 else ("arc_8_16" if m < 16 else "arc_16p")] += 1
                    res["partner_link" if kinds_kept[int(I[j])] == "link" else "partner_section"] += 1
                    ci = int(I[k])
                    if ci not in hcache:
                        hcache[ci] = hops(pred["edges"], len(pred["pts"]), ci)
                    h = hcache[ci].get(int(I[j]), 9)
                    res["hops2" if h == 2 else "hops3p"] += 1
                break


t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    if MAXF and b >= MAXF:
        break
    pr = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                       links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pr, 0.5, 0.5)
    for v in ("c", "a"):
        e = E[v][f.row]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        pairs = make_pairs(e, ia, ib, k, attach=(v == "a"))
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pr["links"]["long"], bool), pairs, snap=True, snap_nogeom=(v == "a"))
        lm = np.asarray(pr["links"]["long"], bool)
        kept_kinds = [kd for i, kd in enumerate(bk["kind"]) if not (kd == "link" and bool(lm[bk["piece_links"][i][0]]) and any(np.linalg.norm(np.asarray(graph.lines[i], float)[0] - q[0]) <= COVER_POS and np.linalg.norm(np.asarray(graph.lines[i], float)[-1] - q[2]) <= COVER_POS for q in pairs))]
        assert len(kept_kinds) == len(raw["pts"]) - n_c, (len(kept_kinds), len(raw["pts"]), n_c)
        run(raw, n_c, kept_kinds, tot[v])
    f._lane = f._ref = None
    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{len(frames)} {time.time() - t0:.0f} s", flush=True)
json.dump(tot, open(os.path.join(ROOT, "temp", "e324", f"dupsrc2_s{S}{'_max' + str(MAXF) if MAXF else ''}.json"), "w"), indent=1)
for v in ("c", "a"):
    print(v, tot[v])

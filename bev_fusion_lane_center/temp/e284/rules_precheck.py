"""scratch (Mirko 2026-11-08, 'work alone, improve the extraction by error analysis and fixes'). From the error analysis of baseline B + the hard close-node step (E275 read-out): the point geometry is decent (tiers 1 + 2: R 0.85, P 0.92 at 1.0 m) but the structure is weak:
topo F1 0.63, junction precision 0.18, dup_frac 0.23, 29 % of the delivered LONG-LINK length overlaps another delivered polyline, pieces_per_gt_piece 1.52. Read-out rules on the output (named output steps, D55: hard selections on the output, no loss through them),
tried on the stored full dump of B (no model), scored by the E273 scorer on all 873 VAL frames, relative to the E275 read-out `close:1.5@0.5` of the same seed:
  N1  long-link NMS: go through the kept LONG links by p descending, drop a long link when >= 70 % of its curve points lie within 1.0 m of the curve of an already kept link (long or short) -> its p is set to 0.4999
  N2  as N1 with >= 50 %
  N3  as N1 with 0.5 m and >= 80 % (strict: only near-identical curves)
  N4  drop a long link when >= 80 % of its curve points lie within 1.0 m of the SECTION polylines (it runs along a lane that is delivered anyway)
  S1  drop SECTION pieces shorter than 3 m that touch no other piece (no edge in the graph)
  S2  the same with 5 m
  N1S1  N1 then S1
PRE-REGISTERED RULE (written before the run; the E275 / E279 bands): a variant is carried to a card when, on BOTH seeds, topo F1 rises by >= 0.020 AND lanes found falls by at most 0.010 AND junction F1 does not fall; a variant that only trades one for the other is not carried. In-sample on VAL 873, a design check, no verdict.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/rules_precheck.py <seed>"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
from lane_task import LaneGraph  # noqa: E402

SEED = int(sys.argv[1])
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
if "--rows" in sys.argv:
    N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1])
    frames, dump = frames[:N_ROWS], dump[:N_ROWS]
VARIANTS = ("base", "N1", "N2", "N3", "N4", "S1", "S2", "N1S1")
KEYS = ("topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "lanes_found", "F1@1.0", "Q", "dup_frac", "vert100_dp")


def long_nms(bk, graph, d, frac, tol, strict_sections=False):
    """ids of the long links to drop"""
    lg = np.asarray(d["link_long"], bool)
    link_pieces = [(i, bk["piece_links"][i][0]) for i, kd in enumerate(bk["kind"]) if kd == "link"]
    longs = [(i, l) for i, l in link_pieces if lg[l]]
    if not longs:
        return []
    order = sorted(longs, key=lambda x: -float(d["link_p"][x[1]]))
    drop = []
    if strict_sections:                                                              # N4: against the section polylines only
        sec = [np.asarray(graph.lines[i], float) for i, kd in enumerate(bk["kind"]) if kd == "section"]
        if not sec:
            return []
        tree = cKDTree(np.concatenate(sec))
        for i, l in order:
            pts = np.asarray(graph.lines[i], float)
            if (tree.query(pts)[0] <= tol).mean() >= frac:
                drop.append(l)
        return drop
    kept_lines = short_links_lines(bk, graph, d)                                     # the short links count as kept curves
    for i, l in order:
        pts = np.asarray(graph.lines[i], float)
        if kept_lines:
            tree = cKDTree(np.concatenate(kept_lines))
            if (tree.query(pts)[0] <= tol).mean() >= frac:
                drop.append(l)
                continue
        kept_lines.append(pts)
    return drop


def short_links_lines(bk, graph, d):
    return [np.asarray(graph.lines[i], float) for i, kd in enumerate(bk["kind"]) if kd == "link" and not bool(np.asarray(d["link_long"])[bk["piece_links"][i][0]])]


def drop_small_sections(graph, bk, min_len):
    deg = np.zeros(len(graph.lines), int)
    for a, b in graph.edges:
        deg[a] += 1
        deg[b] += 1
    keep = []
    for i, l in enumerate(graph.lines):
        ln = float(np.linalg.norm(np.diff(np.asarray(l, float), axis=0), axis=1).sum()) if len(l) > 1 else 0.0
        keep.append(not (bk["kind"][i] == "section" and ln < min_len and deg[i] == 0))
    idx = [i for i, k in enumerate(keep) if k]
    new = {o: n for n, o in enumerate(idx)}
    edges = [(new[a], new[b]) for a, b in graph.edges if a in new and b in new]
    nodes = [graph.nodes[i] for i in idx] if graph.nodes is not None else None
    return LaneGraph([graph.lines[i] for i in idx], edges, True, nodes, None)


def graph_of(pred, p2=None):
    if p2 is not None:
        pred = dict(nodes=pred["nodes"], links=dict(pred["links"], p=p2))
    return QE.readout(pred, 0.5, 0.5)


rec = {v: [] for v in VARIANTS}
t0 = time.time()
n_drop = {v: 0 for v in VARIANTS}
for n, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    graph, bk = graph_of(pred)
    rec["base"].append(QE.graph_counts(graph, f))
    pl = np.asarray(d["link_p"], float)
    drops = {}
    drops["N1"] = long_nms(bk, graph, d, 0.7, 1.0)
    drops["N2"] = long_nms(bk, graph, d, 0.5, 1.0)
    drops["N3"] = long_nms(bk, graph, d, 0.8, 0.5)
    drops["N4"] = long_nms(bk, graph, d, 0.8, 1.0, strict_sections=True)
    for v in ("N1", "N2", "N3", "N4"):
        p2 = pl.copy()
        p2[drops[v]] = 0.4999
        g2, b2 = graph_of(pred, p2)
        rec[v].append(QE.graph_counts(g2, f))
        n_drop[v] += len(drops[v])
    rec["S1"].append(QE.graph_counts(drop_small_sections(graph, bk, 3.0), f))
    rec["S2"].append(QE.graph_counts(drop_small_sections(graph, bk, 5.0), f))
    p2 = pl.copy()
    p2[drops["N1"]] = 0.4999
    g3, b3 = graph_of(pred, p2)
    rec["N1S1"].append(QE.graph_counts(drop_small_sections(g3, b3, 3.0), f))
    f._lane = f._ref = None
    if (n + 1) % 100 == 0:
        print(f"  frame {n + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in VARIANTS}
out = dict(seed=SEED, numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in VARIANTS}, links_dropped=n_drop)
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"rules_precheck_s{SEED}.json"), "w"), indent=1)
print(f"seed {SEED}, {len(frames)} frames")
print(f"{'':7s}" + "".join(f"{k:>12s}" for k in KEYS))
for v in VARIANTS:
    print(f"{v:7s}" + "".join(f"{num[v][k]:12.4f}" if k in num[v] else f"{'-':>12s}" for k in KEYS))
print("differences to base:")
for v in VARIANTS[1:]:
    print(f"{v:7s}" + "".join(f"{num[v][k] - num['base'][k]:+12.4f}" if k in num[v] else f"{'-':>12s}" for k in KEYS))
print("long links dropped:", n_drop)
for v in VARIANTS[1:]:
    ok = (num[v]["topo_F1"] - num["base"]["topo_F1"] >= 0.020) and (num[v]["lanes_found"] - num["base"]["lanes_found"] >= -0.010) and (num[v]["junction_F1"] - num["base"]["junction_F1"] >= 0.0)
    print(f"RULE {v}: {'passed on this seed' if ok else 'not passed'}")

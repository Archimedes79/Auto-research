"""scratch (Mirko 2026-11-10 'Mache weiter' after the card E283 option 5c; D71): does the E273 SCORER reward clean junction ends and good pairs at all? If it does not, an end keypoint head / a pair decision is not worth a GPU hour on scorer evidence.
CPU, VAL 873, baseline B + hard close-node step 1.5 m (E275 read-out) as the delivered graph, one process per seed (argument 0 / 1). A design check, in-sample on VAL, by hand: no verdict on a model. An oracle is a sanity check, not an upper bound.

Variants (the delivered LANE sections are always kept as they are; only the connectors through the junctions change):
  V0  delivered                  the graph of B + hard step as it is
  V1  no connectors              all LONG-link pieces (kind 'link' whose link is long) removed: what the delivered connectors currently give
  V2  GT ends + true pairs       V1 + one cubic Hermite curve (tangent length = chord) per TRUE pair of GT entry and GT exit (the oracle A of oracle_junction.py)
  V3  GT ends + classifier pairs V1 + one curve per candidate pair whose out-of-fold probability is >= 0.5 (HistGradientBoosting on the geometry of the two ends, pair_geometry.py features, 2 folds made of whole physical junctions,
                                 frames linked at 40 m in the same map location)
  V4  GT ends + all candidates   V1 + one curve per geometrically possible pair (the negative control: clean ends, pairs unknown)
An added curve is wired into the graph: an edge from every delivered section piece whose LAST point lies within 4.0 m of the GT entry with its heading within 45 degrees of the entry's, and an edge to every section piece whose FIRST point
lies within 4.0 m of the GT exit likewise (the delivered lane ends are 2 - 4 m off, E283: the wiring tolerance is lenient on purpose); the number of curves wired at both ends is reported. Node ids of a curve: one per GT entry / exit cluster.
Scored by the scorer's own functions (graph_counts / graph_numbers): Q, F1@1.0, lanes found, topo P / R / F1, junction P / R / F1, dup_frac, vert100_dp.

PRE-REGISTERED RULE (written before the run; the bands are those of the E275 / E277 / E279 rule: topo F1 +0.02, junction F1 +0.03):
  THE SCORER REWARDS CLEAN ENDS when V2 beats V0 by more than +0.02 topo F1 AND +0.03 junction F1 on BOTH seeds, with lanes found not lower than V0 - 0.01 and dup_frac not higher than V0 + 0.01.
  REALISTIC POTENTIAL (what an end head + a pair classifier could reach at best): V3 meets the same four conditions against V0 on both seeds AND beats V4 on topo F1 by more than +0.02 (the pairs matter, not the clean ends alone).
  Reading: V2 fails -> the scorer does not reward better connectors (as p16 suggested): no GPU hour for an end head on this evidence; V2 holds and V3 fails -> only perfect pairs help; V2 and V3 hold -> an end head + pair decision is worth a card.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p19_clean_ends_oracle.py <seed>"""
import json
import os
import pickle
import sys
import time

HEAD = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "pair_geometry.py")).read().split("# ---- TRAIN set")[0]
exec(compile(HEAD, "pair_geometry_head", "exec"))          # numpy, cKDTree, LD, QE, QT, HistGradientBoostingClassifier, roc_auc_score, hermite, cluster, allowed, analyse, ROOT
import qgraph_readout as QR  # noqa: E402

SEED = int(sys.argv[1])
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
POSE = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")
t0 = time.time()

# ------------------------------------------------------------------ stage 1: ends and candidate pairs of every frame (seed free)
AN = {}
for b, f in enumerate(frames):
    r = analyse(f.lane())
    if r is not None:
        r.pop("base", None)
        AN[b] = r
    f._lane = f._ref = None
    if (b + 1) % 200 == 0:
        print(f"  analysed {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
fr = sorted(AN)
cen = np.array([(np.array([e[0] for e in AN[b]["ent"]]) @ POSE["R"][frames[b].row].T + POSE["t"][frames[b].row]).mean(0) for b in fr])
loc = np.array([int(POSE["loc"][frames[b].row]) for b in fr])
parent = list(range(len(fr)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for lc in np.unique(loc):
    ix = np.where(loc == lc)[0]
    for a, bb in cKDTree(cen[ix]).query_pairs(40.0):
        ra, rb = find(ix[a]), find(ix[bb])
        if ra != rb:
            parent[ra] = rb
comp = [find(i) for i in range(len(fr))]
ucomp = sorted(set(comp))
fold_of = {c_: i % 2 for i, c_ in enumerate(ucomp)}
fold = {b: fold_of[comp[i]] for i, b in enumerate(fr)}
PROB = {}
oof_y, oof_p = [], []
for k_ in (0, 1):
    tr = [b for b in fr if fold[b] != k_]
    te = [b for b in fr if fold[b] == k_]
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0).fit(np.concatenate([AN[b]["X"] for b in tr]), np.concatenate([AN[b]["y"] for b in tr]))
    for b in te:
        PROB[b] = clf.predict_proba(AN[b]["X"])[:, 1]
        oof_y.append(AN[b]["y"])
        oof_p.append(PROB[b])
oof_y, oof_p = np.concatenate(oof_y), np.concatenate(oof_p)
print(f"junction frames {len(fr)}, physical junctions {len(ucomp)}, candidate pairs {len(oof_y)}, true {int(oof_y.sum())}; out-of-fold AUC {roc_auc_score(oof_y, oof_p):.3f}, "
      f"at 0.5: precision {((oof_p >= 0.5) & (oof_y == 1)).sum() / max((oof_p >= 0.5).sum(), 1):.3f}, recall {((oof_p >= 0.5) & (oof_y == 1)).sum() / oof_y.sum():.3f}", flush=True)


# ------------------------------------------------------------------ stage 2: build the variants and score them
def tan_of(P, at_end):
    q = P[-3:] if at_end else P[:3]
    d = (q[-1] - q[0]) if at_end else (q[-1] - q[0])
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, 1.0])


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def build(lines, edges, nodes, kinds, links, long_mask, r, pairs):
    """V1 (pairs = []) / V2 / V3 / V4: drop the long-link pieces, add one curve per pair; -> raw dict, number of curves, number wired at both ends"""
    keep = [i for i, k in enumerate(kinds) if not (k == "link" and bool(long_mask[links[i][0]]))]
    new = {old: j for j, old in enumerate(keep)}
    L2 = [np.asarray(lines[i], float) for i in keep]
    N2 = [list(nodes[i]) for i in keep]
    E2 = [(new[a], new[b]) for a, b in edges if a in new and b in new]
    ends = [(P[-1], tan_of(P, True), P[0], tan_of(P, False)) for P in L2]
    wired = 0
    for a, b in pairs:
        pa, ta = r["ent"][a]
        pb, tb = r["ext"][b]
        m = len(L2)
        L2.append(hermite(pa, ta, pb, tb, 1.0, 0.25))
        N2.append([10000 + a, 20000 + b])
        w_in = w_out = False
        for i, (le, te, fs, ts) in enumerate(ends):
            if np.linalg.norm(le - pa) <= 4.0 and ang(te, ta) <= 45.0:
                E2.append((i, m))
                w_in = True
            if np.linalg.norm(fs - pb) <= 4.0 and ang(ts, tb) <= 45.0:
                E2.append((m, i))
                w_out = True
        wired += int(w_in and w_out)
    return dict(pts=L2, edges=E2, nodes=N2), len(pairs), wired


dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
VAR = ["V0 delivered", "V1 no connectors", "V2 GT ends + true pairs", "V3 GT ends + classifier pairs", "V4 GT ends + all candidates"]
rec = {v: [] for v in VAR}
added = {v: [0, 0] for v in VAR}
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    rec["V0 delivered"].append(QE.graph_counts(graph, f))
    long_mask = np.asarray(pred["links"]["long"], bool)
    r = AN.get(b)
    pairs = {"V1 no connectors": [], "V2 GT ends + true pairs": [], "V3 GT ends + classifier pairs": [], "V4 GT ends + all candidates": []}
    if r is not None:
        cand = r["cand"]
        pairs["V2 GT ends + true pairs"] = [cand[i] for i in np.flatnonzero(r["y"] == 1)]
        pairs["V3 GT ends + classifier pairs"] = [cand[i] for i in np.flatnonzero(PROB[b] >= 0.5)]
        pairs["V4 GT ends + all candidates"] = list(cand)
    for v, pr in pairs.items():
        raw, n_add, n_wired = build(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, r, pr)
        added[v][0] += n_add
        added[v][1] += n_wired
        rec[v].append(QE.graph_counts(raw, f))
    f._lane = f._ref = None
    if (b + 1) % 100 == 0:
        print(f"  seed {SEED} scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in VAR}
out = dict(seed=SEED, junction_frames=len(fr), oof_auc=float(roc_auc_score(oof_y, oof_p)), curves_added={v: dict(curves=added[v][0], wired_both_ends=added[v][1]) for v in VAR[1:]},
           numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in VAR})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"p19_clean_ends_oracle_s{SEED}.json"), "w"), indent=1)
print(f"seed {SEED}: curves added (wired at both ends): " + ", ".join(f"{v.split(' ')[0]} {added[v][0]} ({added[v][1]})" for v in VAR[1:]))
print(f"{'':36s}" + "".join(f"{k:>12s}" for k in KEYS))
for v in VAR:
    print(f"{v:36s}" + "".join(f"{num[v][k]:12.4f}" for k in KEYS))
print("differences to V0:")
for v in VAR[1:]:
    print(f"{v:36s}" + "".join(f"{num[v][k] - num['V0 delivered'][k]:+12.4f}" for k in KEYS))


def holds(v):
    g = lambda k: num[v][k] - num["V0 delivered"][k]  # noqa: E731
    return g("topo_F1") > 0.02 and g("junction_F1") > 0.03 and g("lanes_found") >= -0.01 and g("dup_frac") <= 0.01


v2, v3 = holds("V2 GT ends + true pairs"), holds("V3 GT ends + classifier pairs")
v3_vs_v4 = num["V3 GT ends + classifier pairs"]["topo_F1"] - num["V4 GT ends + all candidates"]["topo_F1"] > 0.02
print(f"RULE seed {SEED}: V2 meets the four conditions: {v2}; V3 meets them: {v3}; V3 beats V4 on topo F1 by > 0.02: {v3_vs_v4}")

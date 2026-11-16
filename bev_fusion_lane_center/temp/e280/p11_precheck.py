"""E280 P11 pre-check (card experiments/E280_learned-pair-radius-node-step/experiment.md, section 3). CPU only, no model: from the FULL prediction dump of baseline B
(`temp/e276/predfull_e273_p_s<seed>.pkl`, exact map, nodes e >= 0.2) and the lane GT (l4 store) per frame:
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e280/p11_precheck.py <seed> [--rows N] [--auc-only]
Pairs = neighbour cells (3 x 3) with both e >= 0.3, j = the stronger, i = the weaker cell. Type A both cells carry a GT node, B only the stronger, C only the weaker, N neither.
Test 1: AUC (A against B, pairs with d in [1.0, 1.85] m) of gradient boosting on d / (d, e_i, e_j) / all ten features, two folds by scene.
Test 2: the hard greedy step at R = 1.5 m with the oracle exception (a type-A pair is not damped) against plain / hard 1.25 / hard 1.5 (reproduction gate 0.002 against the stored cards).
Output: temp/e280/p11_s<seed>.json (+ p11_pairs_s<seed>.npz for the AUC part). A design check, in-sample on VAL 873, no verdict."""
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as RO  # noqa: E402
import qtargets as QT  # noqa: E402

SEED = int(sys.argv[1])
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
AUC_ONLY = "--auc-only" in sys.argv
OUT = os.path.join(ROOT, "temp", "e280")
TAG = f"s{SEED}" + (f"_rows{N_ROWS}" if N_ROWS else "")
E_PAIR = 0.3
BAND = (1.0, 1.85)
BINS = (0.0, 1.0, 1.25, 1.5, 1.85, 2.5)
HALF = ((0, 1), (1, -1), (1, 0), (1, 1))                      # one of the two directions of each neighbour pair
SPLIT, MERGE = QT.FI["split"], QT.FI["merge"]
TYPES = ("A", "B", "C", "N")
FEATS = ("d", "along", "across", "cos", "e_i", "e_j", "split_i", "merge_i", "split_j", "merge_j")
SETS = {"d": [0], "d,e_i,e_j": [0, 4, 5], "all ten": list(range(10))}
KEYS = ("topo_F1", "junction_F1", "junction_P", "junction_R", "F1@1.0", "lanes_found", "vert100_dp")


def greedy(pos, e, has, radius, oracle, thr=0.5):
    """the hard step of `qgraph_readout.suppress_close_nodes` (e descending, a node stays unless a kept node lies strictly closer than `radius`); oracle: a kept node with a GT node in its
    cell does not damp a node that also has a GT node in its cell (type-A pair)"""
    keep = np.zeros(len(e), bool)
    cand = np.flatnonzero(e >= thr)
    order = cand[np.argsort(-e[cand], kind="stable")]
    kp, kh = np.empty((len(order), 2)), np.empty(len(order), bool)
    n, r2 = 0, float(radius) ** 2
    for i in order:
        if n:
            close = ((kp[:n] - pos[i]) ** 2).sum(1) < r2
            if oracle:
                close &= ~(has[i] & kh[:n])
            if close.any():
                continue
        kp[n], kh[n] = pos[i], has[i]
        n += 1
        keep[i] = True
    return keep


def frame_pairs(d, key, has, gt_sm):
    """-> dict of arrays for the neighbour pairs of one frame (see the module docstring)"""
    e, pos = d["e"], d["pos"]
    idx = np.flatnonzero(e >= E_PAIR)
    cell_node = np.full(2500, -1, int)
    cell_node[key[idx]] = idx
    cx, cy = key[idx] // 50, key[idx] % 50
    A, B = [], []
    for di, dj in HALF:
        nx, ny = cx + di, cy + dj
        ok = (nx >= 0) & (nx < 50) & (ny >= 0) & (ny < 50)
        nb = np.full(len(idx), -1)
        nb[ok] = cell_node[nx[ok] * 50 + ny[ok]]
        s = nb >= 0
        A.append(idx[s])
        B.append(nb[s])
    a, b = np.concatenate(A), np.concatenate(B)
    st = (e[a] > e[b]) | ((e[a] == e[b]) & (a < b))
    j, i = np.where(st, a, b), np.where(st, b, a)
    u = d["dirv"] / np.maximum(np.linalg.norm(d["dirv"], axis=1, keepdims=True), 1e-9)
    dl = pos[j] - pos[i]
    dist = np.linalg.norm(dl, axis=1)
    ui = u[i]
    fl = d["flags"]
    X = np.stack([dist / 2, (dl * ui).sum(1) / 2, (dl[:, 0] * -ui[:, 1] + dl[:, 1] * ui[:, 0]) / 2, (u[i] * u[j]).sum(1), e[i], e[j], fl[i, SPLIT], fl[i, MERGE], fl[j, SPLIT], fl[j, MERGE]], 1)
    hi, hj = has[i], has[j]
    typ = np.where(hi & hj, 0, np.where(hj & ~hi, 1, np.where(hi & ~hj, 2, 3)))
    gtj = gt_sm[i] | gt_sm[j]                                            # a GT node of the pair is a split / merge
    pj = (fl[i, SPLIT] >= 0.5) | (fl[i, MERGE] >= 0.5) | (fl[j, SPLIT] >= 0.5) | (fl[j, MERGE] >= 0.5)
    return dict(dist=dist, typ=typ, X=X, gtj=gtj, pj=pj)


def pass_frames():
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    frames = QE.val_frames()[: len(dump)]
    if N_ROWS:
        dump, frames = dump[:N_ROWS], frames[:N_ROWS]
    names = ("plain", "hard1.25", "hard1.5", "oracle1.5")
    rec = {k: [] for k in names}
    kept_n = {k: 0 for k in names}
    P = {k: [] for k in ("dist", "typ", "X", "gtj", "pj", "scene")}
    t0 = time.time()
    selftested = 0
    for n, (d, f) in enumerate(zip(dump, frames)):
        assert int(d["row"]) == int(f.row), (d["row"], f.row)
        cached = getattr(f, "_ref", None) is not None
        ref = f.ref()
        c = QT.cell_of(d["pos"])
        key = c[:, 0] * 50 + c[:, 1]
        look = np.full(2500, -1, int)
        gc = np.asarray(ref.nt["cell"], int).reshape(-1, 2)
        look[gc[:, 0] * 50 + gc[:, 1]] = np.arange(len(gc))
        has = look[key] >= 0
        gfl = np.asarray(ref.nt["flags"], float).reshape(len(gc), -1)
        gsm = np.zeros(len(key), bool)
        gsm[has] = (gfl[look[key[has]]][:, [SPLIT, MERGE]] > 0.5).any(1)
        pr = frame_pairs(d, key, has, gsm)
        for k, v in pr.items():
            P[k].append(v)
        P["scene"].append(np.full(len(pr["dist"]), f.scene))
        e0 = d["e"]
        if selftested < 5:                                                  # the oracle without any GT node is the plain hard step
            assert (greedy(d["pos"], e0, np.zeros(len(e0), bool), 1.5, True) == RO.suppress_close_nodes(d["pos"], e0, 1.5, 0.5)).all()
            selftested += 1
        keeps = {"plain": e0 >= 0.5,
                 "hard1.25": RO.suppress_close_nodes(d["pos"], e0, 1.25, 0.5),
                 "hard1.5": RO.suppress_close_nodes(d["pos"], e0, 1.5, 0.5),
                 "oracle1.5": greedy(d["pos"], e0, has, 1.5, True)}
        lg = np.asarray(d["link_long"], bool)
        for name, kp in keeps.items():
            kept_n[name] += int(kp.sum())
            nd = dict(pos=d["pos"], dirv=d["dirv"], e=np.where(kp, e0, 0.0), flags=d["flags"], section=d["section"])
            if d.get("s") is not None:
                nd["s"] = d["s"]
            lk = dict(src=d["link_src"], dst=d["link_dst"], p=np.asarray(d["link_p"], float), long=lg, a=d["link_a"], b=d["link_b"])
            graph, _ = QE.readout(dict(nodes=nd, links=lk), 0.5, 0.5)
            rec[name].append(QE.graph_counts(graph, f))
        if not cached:
            f._lane = f._ref = None
        if (n + 1) % 50 == 0:
            print(f"  frame {n + 1}/{len(dump)}  {time.time() - t0:.0f} s", flush=True)
    P = {k: np.concatenate(v) for k, v in P.items()}
    graph = {}
    for name in names:
        g = QE.graph_numbers(rec[name])
        graph[name] = {k: (float(g[k]) if g.get(k) is not None else None) for k in KEYS}
        graph[name]["nodes_delivered"] = kept_n[name]
    return P, graph


def repro(graph):
    """the reproduction gate: plain / hard 1.5 / hard 1.25 against the stored E275 cards (0.002)"""
    ref = {}
    ref["plain"] = json.load(open(os.path.join(ROOT, "runs", "e275", f"e273_p_s{SEED}.plain_t0.5", "card.json")))["p"]["graph"]
    ref["hard1.5"] = json.load(open(os.path.join(ROOT, "runs", "e275", f"e273_p_s{SEED}.close1.5_t0.5", "card.json")))["p"]["graph"]
    ref["hard1.25"] = json.load(open(os.path.join(ROOT, "runs", "e275", f"e273_p_s{SEED}.curve.json")))["curve"]["1.25"]
    out = {}
    for name, r in ref.items():
        out[name] = {k: graph[name][k] - r[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found") if k in r and graph[name][k] is not None}
    return out


def auc_part(P):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    typ, dist, X, scene = P["typ"], P["dist"], P["X"], P["scene"]
    m = (typ <= 1) & (dist >= BAND[0]) & (dist < BAND[1])
    sc = np.unique(scene[m])
    fold_of = {s: k % 2 for k, s in enumerate(sorted(sc))}
    fold = np.array([fold_of.get(s, -1) for s in scene])
    y = (typ == 0).astype(int)
    out = {"pairs_in_band": int(m.sum()), "A_share": float(y[m].mean()) if m.any() else None, "folds": []}
    for tr_fold in (0, 1):
        te_fold = 1 - tr_fold
        tr, te = m & (fold == tr_fold), m & (fold == te_fold)
        res = dict(train_pairs=int(tr.sum()), test_pairs=int(te.sum()), test_A_share=float(y[te].mean()) if te.any() else None)
        if not tr.any() or not te.any() or len(np.unique(y[te])) < 2:       # a subset run with one scene
            res.update({name: None for name in SETS}, gain=None)
            out["folds"].append(res)
            continue
        for name, cols in SETS.items():
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4, min_samples_leaf=40, random_state=0)
            clf.fit(X[tr][:, cols], y[tr])
            res[name] = float(roc_auc_score(y[te], clf.predict_proba(X[te][:, cols])[:, 1]))
        res["gain"] = res["all ten"] - res["d,e_i,e_j"]
        out["folds"].append(res)
    return out


def counts_part(P):
    typ, dist = P["typ"], P["dist"]
    rows = {}
    for b0, b1 in zip(BINS[:-1], BINS[1:]):
        mb = (dist >= b0) & (dist < b1)
        rows[f"{b0:g}-{b1:g}"] = {t: int((mb & (typ == k)).sum()) for k, t in enumerate(TYPES)}
    A = typ == 0
    band = (dist >= BAND[0]) & (dist < BAND[1])
    return dict(by_bin=rows, A_in_band=int((A & band).sum()), A_in_band_gt_split_merge=float(P["gtj"][A & band].mean()) if (A & band).any() else None,
                A_in_band_pred_split_merge=float(P["pj"][A & band].mean()) if (A & band).any() else None,
                B_in_band_pred_split_merge=float(P["pj"][(typ == 1) & band].mean()) if ((typ == 1) & band).any() else None)


if __name__ == "__main__":
    npz = os.path.join(OUT, f"p11_pairs_{TAG}.npz")
    res = dict(seed=SEED, rows=N_ROWS)
    if AUC_ONLY:
        P = dict(np.load(npz))
        old = json.load(open(os.path.join(OUT, f"p11_{TAG}.json")))
        res.update(graph=old.get("graph"), repro=old.get("repro"))
    else:
        P, graph = pass_frames()
        np.savez_compressed(npz, **P)
        res.update(graph=graph, repro=repro(graph))
        try:
            sys.path.insert(0, QE.BP.REPO)
            from autogt import lane_gt
            res["lane_signature"] = lane_gt.signature()
        except Exception as ex:                                              # the signature is a record, not a gate
            res["lane_signature"] = f"unknown ({ex})"
    res["counts"] = counts_part(P)
    res["auc"] = auc_part(P)
    g = res["graph"]
    gate1 = all(f["gain"] is not None and f["gain"] >= 0.05 for f in res["auc"]["folds"])
    gate2 = g["oracle1.5"]["junction_F1"] >= g["hard1.25"]["junction_F1"] + 0.03 and g["oracle1.5"]["lanes_found"] >= g["hard1.25"]["lanes_found"] - 0.005
    res["gate"] = dict(auc_gain_ge_0_05_both_folds=bool(gate1), oracle_ge_hard1_25_plus_0_03_and_lanes=bool(gate2), both=bool(gate1 and gate2))
    json.dump(res, open(os.path.join(OUT, f"p11_{TAG}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

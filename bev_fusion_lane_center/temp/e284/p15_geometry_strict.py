"""scratch (Mirko 2026-11-08, 'work alone'): p14 (geometry of the two ends on the delivered long links) failed its gate, but its label was the E276 pair rule that matches the ends to GT nodes within 2 m, and link_strict.py showed that the ends are 2 - 4 m off along the lane
(only 10 % of the delivered long links are 'right' by that rule, 46 - 52 % are geometrically correct connectors). The label is the problem of p14 as much as the features. RE-TEST with the label fixed beforehand by its definition (link_strict.py): a long link is right when >= 80 % of its
curve points lie within 1.0 m of ONE GT connector run and it runs in the run's direction. Population: the long links of baseline B + the hard close-node step 1.5 m with both ends kept and p >= 0.05 (p15 differs from p14 also in the hard step). The same feature sets and folds as p14
(S1, S1+G, S1p, S1p+G, G alone; folds by scene and by physical junction), AUC pooled and in the decision band p in [0.3, 0.8], precision at recall 0.9.
PRE-REGISTERED RULE (as p14 / E281, written before the run): the gain S1 -> S1+G must be >= +0.05 in both folds (pooled or band) on BOTH seeds, and the same for the folds by physical junction; otherwise the lever is closed.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p15_geometry_strict.py"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

HEAD = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "p14_geometry_links.py")).read().split("OUT = {}")[0]
exec(compile(HEAD, "p14_head", "exec"))          # imports, geometry(), prec_at_recall(), POSES, ROOT
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()


def runs_of(g):
    out = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 4:
            continue
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    out.append(q[k: j + 1])
                k = j + 1
            else:
                k += 1
    return out


OUT = {}
for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    R, Tt, LOC, SCENE = POSES["R"], POSES["t"], POSES["loc"], POSES["scene"]
    rows_ = [int(d["row"]) for d in dump]
    X_S1, X_S1p, X_G, y, scene, frame_id = [], [], [], [], [], []
    cen, loc_f = {}, {}
    for b, (f, d) in enumerate(zip(frames, dump)):
        runs = runs_of(f.lane())
        trees = [cKDTree(r) for r in runs]
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
        keep = QE.kept(pred, 0.5, 0.5)[0]
        pl = np.asarray(d["link_p"], float)
        sel = np.flatnonzero(np.asarray(d["link_long"], bool) & keep[d["link_src"]] & keep[d["link_dst"]] & (d["link_src"] != d["link_dst"]) & (pl >= 0.05))
        f._lane = f._ref = None
        if not len(sel):
            continue
        lab = np.zeros(len(sel), bool)
        for i, l in enumerate(sel):
            a, bb = int(d["link_src"][l]), int(d["link_dst"][l])
            c = QT.hermite(d["pos"][a], d["dirv"][a], d["pos"][bb], d["dirv"][bb], float(d["link_a"][l]), float(d["link_b"][l]), 0.5)
            inb = (np.abs(c) < 48).all(1)
            if not inb.any():
                continue
            for tr in trees:
                dd, idx = tr.query(c)
                if (dd[inb] <= 1.0).mean() >= 0.8 and idx[-1] > idx[0]:
                    lab[i] = True
                    break
        s, t, p = d["link_src"][sel], d["link_dst"][sel], pl[sel]
        L = np.linalg.norm(d["pos"][t] - d["pos"][s], axis=1)
        ds, dt_ = d["dirv"][s], d["dirv"][t]
        dpv = d["pos"][t] - d["pos"][s]
        S1 = np.c_[p, L, d["e"][s], d["e"][t]]
        S1p = np.c_[S1, ds, dt_, d["flags"][s], d["flags"][t], d["link_a"][sel], d["link_b"][sel], (ds * dt_).sum(1), (dpv * ds).sum(1), dpv[:, 0] * ds[:, 1] - dpv[:, 1] * ds[:, 0], d["pos"][s], d["pos"][t]]
        X_S1.append(S1)
        X_S1p.append(S1p)
        X_G.append(geometry(d, sel))
        y.append(lab)
        scene.append(np.full(len(sel), int(SCENE[rows_[b]])))
        frame_id.append(np.full(len(sel), b))
        cen[b], loc_f[b] = d["pos"][np.unique(s)].mean(0) @ R[rows_[b]].T + Tt[rows_[b]], int(LOC[rows_[b]])
        if (b + 1) % 200 == 0:
            print(f"  seed {SEED} frame {b + 1}/{len(frames)}", flush=True)
    S1, S1p, G, y, scene, frame_id = (np.concatenate(a) for a in (X_S1, X_S1p, X_G, y, scene, frame_id))
    scenes = np.unique(scene)
    fold_scene = np.array([{sc: i % 2 for i, sc in enumerate(scenes)}[v] for v in scene])
    fr = sorted(cen)
    C = np.array([cen[b] for b in fr])
    Lc = np.array([loc_f[b] for b in fr])
    parent = list(range(len(fr)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for lc in np.unique(Lc):
        ix = np.where(Lc == lc)[0]
        for a, bb in cKDTree(C[ix]).query_pairs(40.0):
            ra, rb = find(ix[a]), find(ix[bb])
            if ra != rb:
                parent[ra] = rb
    comp = {b: find(i) for i, b in enumerate(fr)}
    ucomp = sorted(set(comp.values()))
    fold_of = {c: i % 2 for i, c in enumerate(ucomp)}
    fold_junction = np.array([fold_of[comp[int(b)]] for b in frame_id])
    print(f"seed {SEED}: candidates {len(y)}, strict right {int(y.sum())} ({100 * y.mean():.1f} %), scenes {len(scenes)}, junction components {len(ucomp)}", flush=True)
    sets = {"S1": S1, "S1+G": np.c_[S1, G], "S1p": S1p, "S1p+G": np.c_[S1p, G], "G alone": G}
    band = (S1[:, 0] >= 0.3) & (S1[:, 0] <= 0.8)
    res_seed = {}
    for fold_name, fold in (("folds by scene", fold_scene), ("folds by physical junction", fold_junction)):
        pred_ = {k: np.full(len(y), np.nan) for k in sets}
        for k_ in (0, 1):
            tr, te = (fold != k_), (fold == k_)
            for name, F in sets.items():
                clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0).fit(F[tr], y[tr])
                pred_[name][te] = clf.predict_proba(F[te])[:, 1]
        out = {}
        for name in sets:
            out[name] = dict(pooled=[float(roc_auc_score(y[fold == k_], pred_[name][fold == k_])) for k_ in (0, 1)], band=[float(roc_auc_score(y[band & (fold == k_)], pred_[name][band & (fold == k_)])) for k_ in (0, 1)])
            pr, kept = prec_at_recall(y, pred_[name], 0.9)
            out[name]["precision_at_recall_0.9"], out[name]["kept_share_at_recall_0.9"] = pr, kept
        for lo, hi in (("S1", "S1+G"), ("S1p", "S1p+G")):
            out[f"gain {lo} -> {hi}"] = dict(pooled=[out[hi]["pooled"][k_] - out[lo]["pooled"][k_] for k_ in (0, 1)], band=[out[hi]["band"][k_] - out[lo]["band"][k_] for k_ in (0, 1)])
        res_seed[fold_name] = out
        print(f"  [{fold_name}] n {len(y)}")
        for name in sets:
            o = out[name]
            print(f"      {name:8s} AUC pooled {o['pooled'][0]:.3f} / {o['pooled'][1]:.3f}   band {o['band'][0]:.3f} / {o['band'][1]:.3f}   precision at recall 0.9: {o['precision_at_recall_0.9']:.3f} (keeps {100 * o['kept_share_at_recall_0.9']:.0f} %)")
        for g_ in ("gain S1 -> S1+G", "gain S1p -> S1p+G"):
            print(f"      {g_}: pooled {out[g_]['pooled'][0]:+.3f} / {out[g_]['pooled'][1]:+.3f}   band {out[g_]['band'][0]:+.3f} / {out[g_]['band'][1]:+.3f}")
    OUT[f"seed {SEED}"] = res_seed
json.dump(OUT, open(os.path.join(ROOT, "temp", "e284", "p15_geometry_strict.json"), "w"), indent=1)
print("written p15_geometry_strict.json")

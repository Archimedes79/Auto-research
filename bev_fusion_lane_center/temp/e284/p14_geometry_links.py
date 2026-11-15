"""scratch (Mirko's OK of 2026-11-08 'Ja, messen' after pair_geometry.py / pair_loo.py): does the GEOMETRY of the two ends tell right from wrong among the delivered long links of baseline B? CPU, no model, in the style of the E281 pre-check
(temp/e281/p12_precheck.py, Part B): the FULL prediction dump of B (`temp/e276/predfull_e273_p_s<seed>.pkl`), the cached pair-rule labels (`temp/e281/info_s<seed>.pkl`: both / matched / right), VAL 873, long links with both ends delivered (e >= 0.5).
Feature sets: S1 = (p, length, e_u, e_v) as in E281; S1+G = S1 + the geometry of the pair from the DELIVERED nodes (turn angle between the two node headings, chord in the entry's and the exit's frame, chord angles, the entry's / exit's lane rank among the
delivered long-link ends within 12 m with a similar heading, candidates per entry / exit, rank of the pair by turn angle and chord among them, numbers of entries and exits); S1p = E281's strong current-frame set (headings, flags, link scales, positions), S1p+G; G alone.
Folds: (a) two folds by scene (as E281), (b) two folds by PHYSICAL JUNCTION (frames linked when their junction centres lie within 40 m at the same map location; components alternate over the folds): a gain that survives (b) is not a memorised layout.
AUC of 'right' pooled and in the decision band p in [0.3, 0.8]; the gate of E281: gain >= +0.05 in both folds. Also: the precision of the kept links at recall 0.9 (pooled out-of-fold predictions).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash tools/env/_py14.sh temp/e284/p14_geometry_links.py"""
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.environ["BFLC_ROOT"]
THR = 0.5
POSES = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))


def sang(u, v):
    return np.degrees(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1)))


def geometry(d, sel):
    s, t = d["link_src"][sel], d["link_dst"][sel]
    pos = d["pos"]
    u = d["dirv"] / (np.linalg.norm(d["dirv"], axis=1, keepdims=True) + 1e-9)
    ua, ub, pa, pb = u[s], u[t], pos[s], pos[t]
    ch = pb - pa
    L = np.linalg.norm(ch, axis=1)
    chn = ch / (L[:, None] + 1e-9)
    na, nb = np.c_[-ua[:, 1], ua[:, 0]], np.c_[-ub[:, 1], ub[:, 0]]
    turn = sang(ua, ub)

    def side(nodes):
        out = {}
        P, U = pos[nodes], u[nodes]
        for i, n in enumerate(nodes):
            dd = P - P[i]
            ang = np.degrees(np.arccos(np.clip(U @ U[i], -1, 1)))
            m = (np.linalg.norm(dd, axis=1) <= 12.0) & (ang <= 30.0)
            m[i] = False
            lat = dd[m] @ np.array([-U[i][1], U[i][0]])
            out[int(n)] = (int((lat > 1.0).sum()), int((lat < -1.0).sum()))
        return out

    src_nodes, dst_nodes = np.unique(s), np.unique(t)
    sa, sb = side(src_nodes), side(dst_nodes)
    cs, ct = Counter(s.tolist()), Counter(t.tolist())
    rk_turn_a, rk_len_a, rk_turn_b = np.zeros(len(s)), np.zeros(len(s)), np.zeros(len(s))
    for node in src_nodes:
        idx = np.flatnonzero(s == node)
        rk_turn_a[idx] = np.argsort(np.argsort(np.abs(turn[idx]))) / max(len(idx) - 1, 1)
        rk_len_a[idx] = np.argsort(np.argsort(L[idx])) / max(len(idx) - 1, 1)
    for node in dst_nodes:
        idx = np.flatnonzero(t == node)
        rk_turn_b[idx] = np.argsort(np.argsort(np.abs(turn[idx]))) / max(len(idx) - 1, 1)
    G = np.c_[turn, np.abs(turn), L, (ch * ua).sum(1), (ch * na).sum(1), (ch * ub).sum(1), (ch * nb).sum(1), sang(ua, chn), sang(ub, chn),
              [sa[int(x)][0] for x in s], [sa[int(x)][1] for x in s], [sb[int(x)][0] for x in t], [sb[int(x)][1] for x in t],
              [cs[int(x)] for x in s], [ct[int(x)] for x in t], rk_turn_a, rk_len_a, rk_turn_b, np.full(len(s), len(src_nodes)), np.full(len(s), len(dst_nodes))]
    return G


def prec_at_recall(y, p, target=0.9):
    for th in np.sort(np.unique(p))[::-1]:
        keep = p >= th
        if (keep & (y == 1)).sum() / max(y.sum(), 1) >= target:
            return float((keep & (y == 1)).sum() / keep.sum()), float(keep.sum() / len(y))
    return None, None


OUT = {}
for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    INFO = pickle.load(open(os.path.join(ROOT, "temp", "e281", f"info_s{SEED}.pkl"), "rb"))
    rows_ = [int(d["row"]) for d in dump]
    R, Tt, LOC, SCENE = POSES["R"], POSES["t"], POSES["loc"], POSES["scene"]
    X_S1, X_S1p, X_G, y, matched, scene, frame_id = [], [], [], [], [], [], []
    cen, loc_f = {}, {}
    for b, d in enumerate(dump):
        I = INFO[b]
        sel = np.flatnonzero(np.asarray(d["link_long"], bool) & I["both"])
        if not len(sel):
            continue
        s, t, p = d["link_src"][sel], d["link_dst"][sel], d["link_p"][sel]
        L = np.linalg.norm(d["pos"][t] - d["pos"][s], axis=1)
        ds, dt_ = d["dirv"][s], d["dirv"][t]
        dpv = d["pos"][t] - d["pos"][s]
        S1 = np.c_[p, L, d["e"][s], d["e"][t]]
        S1p = np.c_[S1, ds, dt_, d["flags"][s], d["flags"][t], d["link_a"][sel], d["link_b"][sel], (ds * dt_).sum(1), (dpv * ds).sum(1), dpv[:, 0] * ds[:, 1] - dpv[:, 1] * ds[:, 0], d["pos"][s], d["pos"][t]]
        X_S1.append(S1)
        X_S1p.append(S1p)
        X_G.append(geometry(d, sel))
        y.append(I["right"][sel])
        matched.append(I["matched"][sel])
        scene.append(np.full(len(sel), int(SCENE[rows_[b]])))
        frame_id.append(np.full(len(sel), b))
        gp = d["pos"][np.unique(s)].mean(0) @ R[rows_[b]].T + Tt[rows_[b]]
        cen[b], loc_f[b] = gp, int(LOC[rows_[b]])
    S1, S1p, G, y, matched, scene, frame_id = (np.concatenate(a) for a in (X_S1, X_S1p, X_G, y, matched, scene, frame_id))
    # folds: (a) by scene, (b) by physical junction
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
    print(f"seed {SEED}: candidates {len(y)} (matched {int(matched.sum())}), right {int(y.sum())}, scenes {len(scenes)}, junction components {len(ucomp)}", flush=True)

    sets = {"S1": S1, "S1+G": np.c_[S1, G], "S1p": S1p, "S1p+G": np.c_[S1p, G], "G alone": G}
    band = (S1[:, 0] >= 0.3) & (S1[:, 0] <= 0.8)
    res_seed = {}
    for label_name, mask in (("matched ends, right vs wrong (card label)", matched), ("every candidate, right vs the rest", np.ones(len(y), bool))):
        res_seed[label_name] = {}
        for fold_name, fold in (("folds by scene", fold_scene), ("folds by physical junction", fold_junction)):
            pred = {k: np.full(len(y), np.nan) for k in sets}
            for k_ in (0, 1):
                tr, te = mask & (fold != k_), mask & (fold == k_)
                for name, F in sets.items():
                    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0).fit(F[tr], y[tr])
                    pred[name][te] = clf.predict_proba(F[te])[:, 1]
            out = {}
            for name in sets:
                out[name] = dict(pooled=[float(roc_auc_score(y[mask & (fold == k_)], pred[name][mask & (fold == k_)])) for k_ in (0, 1)],
                                 band=[float(roc_auc_score(y[mask & band & (fold == k_)], pred[name][mask & band & (fold == k_)])) for k_ in (0, 1)])
                pr, kept = prec_at_recall(y[mask], pred[name][mask], 0.9)
                out[name]["precision_at_recall_0.9"], out[name]["kept_share_at_recall_0.9"] = pr, kept
            for lo, hi in (("S1", "S1+G"), ("S1p", "S1p+G")):
                out[f"gain {lo} -> {hi}"] = dict(pooled=[out[hi]["pooled"][k_] - out[lo]["pooled"][k_] for k_ in (0, 1)], band=[out[hi]["band"][k_] - out[lo]["band"][k_] for k_ in (0, 1)])
            out["n"], out["pos"] = int(mask.sum()), int(y[mask].sum())
            res_seed[label_name][fold_name] = out
            print(f"  [{label_name}] [{fold_name}] n {out['n']} pos {out['pos']}")
            for name in sets:
                o = out[name]
                print(f"      {name:8s} AUC pooled {o['pooled'][0]:.3f} / {o['pooled'][1]:.3f}   band {o['band'][0]:.3f} / {o['band'][1]:.3f}   precision at recall 0.9: {o['precision_at_recall_0.9']:.3f} (keeps {100 * o['kept_share_at_recall_0.9']:.0f} % of the candidates)")
            for g in ("gain S1 -> S1+G", "gain S1p -> S1p+G"):
                print(f"      {g}: pooled {out[g]['pooled'][0]:+.3f} / {out[g]['pooled'][1]:+.3f}   band {out[g]['band'][0]:+.3f} / {out[g]['band'][1]:+.3f}")
    OUT[f"seed {SEED}"] = res_seed
json.dump(OUT, open(os.path.join(ROOT, "temp", "e284", "p14_geometry_links.json"), "w"), indent=1)
print("written p14_geometry_links.json")

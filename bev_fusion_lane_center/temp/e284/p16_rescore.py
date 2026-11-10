"""scratch (Mirko 2026-11-08 / 09, 'work alone, improve the extraction'): p15 (strict curve-level label, geometry of the two ends on the delivered long links) PASSED its pre-registered gate: gain S1 -> S1+G >= +0.05 in both folds, both seeds, both fold schemes
(pooled +0.06 ... +0.10, band +0.09 ... +0.14); against the strong current-frame set S1p the geometry adds only +0.01 ... +0.03 AUC. Does a RE-SCORED read-out of the long links help the delivered graph, as the E273 scorer sees it?
Out-of-fold (2 folds by PHYSICAL JUNCTION, as p15) probabilities p' of 'strict right' for every long-link candidate (kept ends, p >= 0.05) replace the head's p in the read-out of baseline B + the hard close-node step 1.5 m (E275): a long link is delivered when p' >= tau', tau' chosen so that
the NUMBER of delivered long links equals that of the base read-out (p >= 0.5) on the same seed (count-matched: the scorer cannot be fooled by simply delivering more or less); all other links are as delivered. Feature sets: S1p (the head's own evidence + headings, flags, scales, positions: the strong baseline), S1+G, S1p+G. Variants:
  C0  base (p >= 0.5)               C1  count-matched with p' of S1p (the control: a better use of the head's own features, no geometry)
  C2  count-matched with S1+G       C3  count-matched with S1p+G         C4  p' (S1p+G) >= 0.40
PRE-REGISTERED RULE (written before the run): the geometry is carried to a card when C2 or C3 raises topo F1 by >= 0.020 over C0 on BOTH seeds, with lanes found falling by at most 0.010 and junction F1 not falling, AND beats the control C1 on topo F1 by >= 0.010 on both seeds (so the geometry, not the re-fit, brings it).
Scored on all 873 VAL frames; in-sample on VAL in the sense that the 2-fold split is by junction, not by train / test: a design check, no verdict.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p16_rescore.py <seed>"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HEAD = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "p14_geometry_links.py")).read().split("OUT = {}")[0]
exec(compile(HEAD, "p14_head", "exec"))          # imports, geometry(), POSES, ROOT, HistGradientBoostingClassifier
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SEED = int(sys.argv[1])
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
R, Tt, LOC, SCENE = POSES["R"], POSES["t"], POSES["loc"], POSES["scene"]
rows_ = [int(d["row"]) for d in dump]
KEYS = ("topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "lanes_found", "F1@1.0", "Q", "dup_frac")


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


preds, sels, labs, X_S1, X_S1p, X_G, frame_id = {}, {}, [], [], [], [], []
cen, loc_f = {}, {}
t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    runs = runs_of(f.lane())
    trees = [cKDTree(r) for r in runs]
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
    keep = QE.kept(pred, 0.5, 0.5)[0]
    pl = np.asarray(d["link_p"], float)
    sel = np.flatnonzero(np.asarray(d["link_long"], bool) & keep[d["link_src"]] & keep[d["link_dst"]] & (d["link_src"] != d["link_dst"]) & (pl >= 0.05))
    preds[b] = pred
    sels[b] = sel
    if not len(sel):
        f._lane = f._ref = None
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
    labs.append(lab)
    frame_id.append(np.full(len(sel), b))
    cen[b], loc_f[b] = d["pos"][np.unique(s)].mean(0) @ R[rows_[b]].T + Tt[rows_[b]], int(LOC[rows_[b]])
    f._lane = f._ref = None
    if (b + 1) % 200 == 0:
        print(f"  seed {SEED} labelled {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
S1, S1p, G, y, frame_id = (np.concatenate(a) for a in (X_S1, X_S1p, X_G, labs, frame_id))
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
fold = np.array([fold_of[comp[int(b)]] for b in frame_id])
sets = {"S1p": S1p, "S1+G": np.c_[S1, G], "S1p+G": np.c_[S1p, G]}
pp = {k: np.zeros(len(y)) for k in sets}
for k_ in (0, 1):
    for name, F in sets.items():
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0).fit(F[fold != k_], y[fold != k_])
        pp[name][fold == k_] = clf.predict_proba(F[fold == k_])[:, 1]
n_base = int((S1[:, 0] >= 0.5).sum())
taus = {name: float(np.sort(pp[name])[::-1][n_base - 1]) for name in sets}
print(f"seed {SEED}: candidates {len(y)}, strict right {100 * y.mean():.1f} %, base delivers {n_base} long links; count-matched tau' " + ", ".join(f"{k} {v:.3f}" for k, v in taus.items()), flush=True)
VAR = {"C0 base": None, "C1 count-matched, S1p (control)": ("S1p", taus["S1p"]), "C2 count-matched, S1+G": ("S1+G", taus["S1+G"]), "C3 count-matched, S1p+G": ("S1p+G", taus["S1p+G"]), "C4 S1p+G >= 0.40": ("S1p+G", 0.40)}
off = {}
pos_ = 0
for b in fr:
    off[b] = (pos_, pos_ + len(sels[b]))
    pos_ += len(sels[b])
rec = {v: [] for v in VAR}
delivered = {v: 0 for v in VAR}
precision = {v: [0, 0] for v in VAR}
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = preds[b]
    pl = np.asarray(d["link_p"], float)
    for v, spec in VAR.items():
        if spec is None or b not in off:
            p2 = pl
        else:
            name, tau = spec
            a0, a1 = off[b]
            pr = pp[name][a0:a1]
            p2 = pl.copy()
            p2[sels[b]] = np.where(pr >= tau, np.maximum(pl[sels[b]], 0.5), np.minimum(pl[sels[b]], 0.4999))
            if spec is not None:
                ok = pr >= tau
                precision[v][0] += int((y[a0:a1] & ok).sum())
                precision[v][1] += int(ok.sum())
        graph, bk = QE.readout(dict(nodes=pred["nodes"], links=dict(pred["links"], p=p2)), 0.5, 0.5)
        rec[v].append(QE.graph_counts(graph, f))
    if spec is None:
        pass
    f._lane = f._ref = None
    if (b + 1) % 200 == 0:
        print(f"  seed {SEED} scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in VAR}
out = dict(seed=SEED, n_base=n_base, taus=taus, numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in VAR},
           strict_precision_of_delivered={v: (precision[v][0] / max(precision[v][1], 1) if precision[v][1] else float(y[S1[:, 0] >= 0.5].mean())) for v in VAR})
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"p16_rescore_s{SEED}.json"), "w"), indent=1)
print(f"seed {SEED}: strict precision of the delivered long links: " + ", ".join(f"{v.split(' ')[0]} {100 * out['strict_precision_of_delivered'][v]:.1f} %" for v in VAR))
print(f"{'':36s}" + "".join(f"{k:>12s}" for k in KEYS))
for v in VAR:
    print(f"{v:36s}" + "".join(f"{num[v][k]:12.4f}" for k in KEYS))
print("differences to C0:")
for v in list(VAR)[1:]:
    print(f"{v:36s}" + "".join(f"{num[v][k] - num['C0 base'][k]:+12.4f}" for k in KEYS))
g = lambda v, k: num[v][k] - num["C0 base"][k]  # noqa: E731
for v in ("C2 count-matched, S1+G", "C3 count-matched, S1p+G"):
    ok = g(v, "topo_F1") >= 0.020 and g(v, "lanes_found") >= -0.010 and g(v, "junction_F1") >= 0.0 and (num[v]["topo_F1"] - num["C1 count-matched, S1p (control)"]["topo_F1"]) >= 0.010
    print(f"RULE {v}: {'passed on this seed' if ok else 'not passed'}")

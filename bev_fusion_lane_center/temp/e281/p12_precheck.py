"""E281 P12 pre-check (card experiments/E281_long-link-memory/experiment.md, section 3). CPU only, no model: from the FULL prediction dump of baseline B (`temp/e276/predfull_e273_p_s<seed>.pkl`, exact map,
nodes e >= 0.2, links p >= 0.05), the poses of the VAL file and the lane GT (l4 store), in time order:
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e281/p12_precheck.py <seed> [--rows N]
Built from the read-only rehearsal scripts of the critic pass (q1 counts / persistence, q2 test 1, q3 test 2), with the scorer added.
Part A  counts and persistence of the delivered long links by pair-rule label and by cause of non-persistence (E276 P5b / P9 definitions).
Part B  test 1: AUC for 'right' (pair rule, matched ends) of gradient boosting on S1 = (p, length, e_u, e_v) against S2 = S1 + memory (m1: highest probability of a previous long link with both warped
        ends within 2 m; m2, m3: existence probability of the nearest previous node within 2 m of each end), two folds by scene, pooled and in the decision band p in [0.3, 0.8]; reported: a strong current-frame baseline, the label 'right against everything else'.
Part C  test 2: hand-made read-outs V1-V6 of the long-link probability from the current and the previous frame, scored by the E273 scorer (topo F1, junction F1, F1@1.0, lanes found), the pair-rule precision of the
        delivered long links and long_persist. The variant 'none' must reproduce the stored E275 card of B.
Output: temp/e281/p12_s<seed>.json. A design check, in-sample on VAL 873, no verdict."""
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
from lane_task import measures as MS  # noqa: E402

SEED = int(sys.argv[1])
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
OUT = os.path.join(ROOT, "temp", "e281")
TAG = f"s{SEED}" + (f"_rows{N_ROWS}" if N_ROWS else "")
THR, TOL, BOXM = 0.5, 2.0, 45.0
VARIANTS = ("none", "V1 average", "V2 filter prev>=0.3", "V3 hold p>=0.25 prev>=0.5", "V4 hold p>=0.1 prev>=0.5", "V5 filter prev>=0.5", "V6 recursive average")
POSES = os.path.expanduser("~/exp_gcar_b123/poses_val.npz")

frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
assert [d["row"] for d in dump] == [f.row for f in frames]
if N_ROWS:
    frames, dump = frames[:N_ROWS], dump[:N_ROWS]
rows = [f.row for f in frames]
by_row = {r: i for i, r in enumerate(rows)}
poses = np.load(POSES)
T, R, PREV, SCENE = poses["t"], poses["R"], poses["prev"], poses["scene"]
prev_of = {by_row[r]: by_row[int(PREV[r])] for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row}
order = sorted(range(len(frames)), key=lambda i: rows[i])                       # time order inside a scene (prev has the smaller row)
res = dict(seed=SEED, rows=N_ROWS, frames=len(frames), pairs=len(prev_of), poses=dict(path=POSES, size=os.path.getsize(POSES), mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(POSES)))))


def lane2ego(p):
    return np.c_[p[:, 1], -p[:, 0]]


def ego2lane(e):
    return np.c_[-e[:, 1], e[:, 0]]


CONV = {"ego pose, lane frame converted": lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra.T + ta - tb) @ rb),
        "pose acts on the lane frame": lambda p, ra, ta, rb, tb: (p @ ra.T + ta - tb) @ rb,
        "ego pose, R transposed": lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra + ta - tb) @ rb.T)}


def warp_with(conv, p, a, b):
    return CONV[conv](np.asarray(p, float), R[rows[a]], T[rows[a]], R[rows[b]], T[rows[b]])


# ---- the convention of the pose file, from the GT nodes of up to 40 pairs (as exp_temporal_links.py)
pl = [(a, b) for b, a in prev_of.items()]
test = pl[:: max(1, len(pl) // 40)][:40]
score = {c: [] for c in CONV}
for a, b in test:
    Ga, Gb = frames[a].ref().nt["pos"], frames[b].ref().nt["pos"]
    if not len(Ga) or not len(Gb):
        continue
    tree = cKDTree(Gb)
    for c in CONV:
        w = warp_with(c, Ga, a, b)
        m = (np.abs(w) < BOXM).all(1)
        if m.any():
            score[c].append(float(np.median(tree.query(w[m])[0])))
    frames[a]._lane = frames[a]._ref = frames[b]._lane = frames[b]._ref = None
best = min(score, key=lambda c: np.median(score[c]))
assert np.median(score[best]) < 1.5, "no convention of the pose file aligns the GT nodes"
res["convention"] = dict(chosen=best, median_alignment_m={c: round(float(np.median(v)), 2) for c, v in score.items()})
print("convention:", res["convention"], flush=True)


def warp(p, a, b):
    return warp_with(best, p, a, b)


# ---- labels (the pair rule of E276 P9) per frame, cached
cache = os.path.join(OUT, f"info_{TAG}.pkl")
if os.path.exists(cache):
    INFO = pickle.load(open(cache, "rb"))
else:
    INFO = []
    for f, d in zip(frames, dump):
        cached = getattr(f, "_ref", None) is not None
        ref = f.ref()
        keep = d["e"] >= THR
        kidx = np.flatnonzero(keep)
        P, G, nt = d["pos"][keep], ref.nt["pos"], ref.nt
        pr = MS.match_1to1(P, G, QE.NODE_TOL) if len(P) and len(G) else np.zeros((0, 2), int)
        g_of = -np.ones(len(d["e"]), int)
        for p_, gi in pr:
            g_of[kidx[p_]] = gi
        gt_link = {(int(a), int(b)) for a, b in zip(nt["src"], nt["dst"])}
        s, t = d["link_src"], d["link_dst"]
        both = keep[s] & keep[t] & (s != t)
        matched = both & (g_of[s] >= 0) & (g_of[t] >= 0)
        right = np.array([(int(g_of[a]), int(g_of[b])) in gt_link if m else False for a, b, m in zip(s, t, matched)], bool)
        INFO.append(dict(both=both, matched=matched, right=right))
        if not cached:
            f._lane = f._ref = None
    pickle.dump(INFO, open(cache, "wb"))
print("labels ready", flush=True)


def match_prev(b, a, vals_a):
    """for every link of frame b: the highest value among the long links of the previous frame a whose two ends, moved into the frame of b, lie within TOL of the two ends (0 if none); also the
    nearest-previous-node existence probability at each end"""
    db, da = dump[b], dump[a]
    n = len(db["link_p"])
    m1, m2, m3 = np.zeros(n), np.zeros(n), np.zeros(n)
    pa = warp(da["pos"], a, b)
    lgp = np.asarray(da["link_long"], bool)
    Ws, Wt, vv = pa[da["link_src"][lgp]], pa[da["link_dst"][lgp]], vals_a[lgp]
    sel = np.flatnonzero(np.asarray(db["link_long"], bool))
    if len(Ws) and len(sel):
        cand = cKDTree(Ws).query_ball_point(db["pos"][db["link_src"][sel]], TOL)
        for i, c in zip(sel, cand):
            if c:
                c = np.asarray(c)
                ok = np.linalg.norm(Wt[c] - db["pos"][db["link_dst"][i]], axis=1) <= TOL
                if ok.any():
                    m1[i] = vv[c[ok]].max()
    if len(sel):
        tn = cKDTree(pa)
        ds, is_ = tn.query(db["pos"][db["link_src"][sel]])
        dt, it = tn.query(db["pos"][db["link_dst"][sel]])
        m2[sel] = np.where(ds <= TOL, da["e"][is_], 0.0)
        m3[sel] = np.where(dt <= TOL, da["e"][it], 0.0)
    return m1, m2, m3


# ---- raw memory features for every frame with a previous frame (previous raw probabilities, no recursion)
MEM = {b: match_prev(b, a, dump[a]["link_p"]) for b, a in prev_of.items()}
print("memory features ready", flush=True)


def persist_one(A, B, C, D):
    hit = np.zeros(len(A), bool)
    if not len(C):
        return hit
    tc = cKDTree(C)
    for i, cand in enumerate(tc.query_ball_point(A, TOL)):
        if cand and (np.linalg.norm(D[cand] - B[i], axis=1) <= TOL).any():
            hit[i] = True
    return hit


def delivered(d, dm_long):
    """links of a frame delivered: both ends e >= 0.5 and (long links: the variant's mask, short links: p >= 0.5)"""
    keep = d["e"] >= THR
    s, t, p, lg = d["link_src"], d["link_dst"], d["link_p"], np.asarray(d["link_long"], bool)
    return keep[s] & keep[t] & (s != t) & np.where(lg, dm_long, p >= THR)


def persistence(dl, detail=None):
    h = n = 0
    for b, a in prev_of.items():
        da, db = dump[a], dump[b]
        ma = dl[a] & np.asarray(da["link_long"], bool)
        A, B = warp(da["pos"][da["link_src"][ma]], a, b), warp(da["pos"][da["link_dst"][ma]], a, b)
        inb = (np.abs(A) < BOXM).all(1) & (np.abs(B) < BOXM).all(1)
        A, B = A[inb], B[inb]
        mb = dl[b]
        hit = persist_one(A, B, db["pos"][db["link_src"][mb]], db["pos"][db["link_dst"][mb]])
        h += int(hit.sum())
        n += len(hit)
        if detail is not None:
            I = INFO[a]
            lab = np.where(I["right"][ma][inb], "right", np.where(I["matched"][ma][inb], "wrong", "unmatched"))
            for l in ("right", "wrong", "unmatched"):
                detail.setdefault(l, np.zeros(2))
                detail[l] += (int(hit[lab == l].sum()), int((lab == l).sum()))
    return h / max(n, 1), n


# ================================================================ Part A
lgs = [np.asarray(d["link_long"], bool) for d in dump]
dl0 = [delivered(d, d["link_p"] >= THR) for d in dump]
nd = sum(int((dl0[i] & lgs[i]).sum()) for i in range(len(dump)))
nm = sum(int((dl0[i] & lgs[i] & INFO[i]["matched"]).sum()) for i in range(len(dump)))
nr = sum(int((dl0[i] & lgs[i] & INFO[i]["right"]).sum()) for i in range(len(dump)))
lens = np.concatenate([np.linalg.norm(d["pos"][d["link_dst"][dl0[i] & lgs[i]]] - d["pos"][d["link_src"][dl0[i] & lgs[i]]], axis=1) for i, d in enumerate(dump)])
detail = {}
pers, npers = persistence(dl0, detail)
res["A"] = dict(delivered_long=nd, matched=nm, right=nr, precision_matched=nr / max(nm, 1), share_len_le_11_4=float((lens <= 11.4).mean()), persistence=pers, persistence_n=npers,
                persistence_by_label={k: dict(persist=float(v[0] / max(v[1], 1)), n=int(v[1])) for k, v in detail.items()})
print("A:", json.dumps(res["A"]), flush=True)

# ================================================================ Part B
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rowsX, scene, fr = [], [], []
    for b, a in prev_of.items():
        d, I = dump[b], INFO[b]
        sel = np.flatnonzero(lgs[b] & I["both"])
        if not len(sel):
            continue
        s, t, p = d["link_src"][sel], d["link_dst"][sel], d["link_p"][sel]
        m1, m2, m3 = (m[sel] for m in MEM[b])
        L = np.linalg.norm(d["pos"][t] - d["pos"][s], axis=1)
        ds, dt_ = d["dirv"][s], d["dirv"][t]
        dpv = d["pos"][t] - d["pos"][s]
        S1 = np.c_[p, L, d["e"][s], d["e"][t]]
        S1p = np.c_[S1, ds, dt_, d["flags"][s], d["flags"][t], d["link_a"][sel], d["link_b"][sel], (ds * dt_).sum(1), (dpv * ds).sum(1), dpv[:, 0] * ds[:, 1] - dpv[:, 1] * ds[:, 0], d["pos"][s], d["pos"][t]]
        rowsX.append((S1, S1p, np.c_[m1, m2, m3], I["right"][sel], I["matched"][sel], np.full(len(sel), int(SCENE[rows[b]]))))
    S1, S1p, MM, y, matched, scene = (np.concatenate([r[i] for r in rowsX]) for i in range(6))
    scenes = np.unique(scene)
    fold = np.array([{sc: i % 2 for i, sc in enumerate(scenes)}[s] for s in scene])

    def oof(F, yy, mask):
        pred = np.full(len(yy), np.nan)
        for k in (0, 1):
            tr, te = mask & (fold != k), mask & (fold == k)
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
            clf.fit(F[tr], yy[tr])
            pred[te] = clf.predict_proba(F[te])[:, 1]
        return pred

    def auc_table(name, mask, yy):
        sets = {"S1": S1, "S2": np.c_[S1, MM], "S1+ (strong current)": S1p, "S2+ (strong + memory)": np.c_[S1p, MM]}
        pr = {k: oof(F, yy, mask) for k, F in sets.items()}
        band = (S1[:, 0] >= 0.3) & (S1[:, 0] <= 0.8)
        out = dict(n=int(mask.sum()), pos=int(yy[mask].sum()), band_n=int((mask & band).sum()), band_pos=int(yy[mask & band].sum()))
        for k in sets:
            out[k] = dict(pooled=[float(roc_auc_score(yy[mask & (fold == f)], pr[k][mask & (fold == f)])) for f in (0, 1)],
                          band=[float(roc_auc_score(yy[mask & band & (fold == f)], pr[k][mask & band & (fold == f)])) for f in (0, 1)])
        for lo, hi in (("S1", "S2"), ("S1+ (strong current)", "S2+ (strong + memory)")):
            out[f"gain {lo} -> {hi}"] = dict(pooled=[out[hi]["pooled"][f] - out[lo]["pooled"][f] for f in (0, 1)], band=[out[hi]["band"][f] - out[lo]["band"][f] for f in (0, 1)])
        return out

    res["B"] = dict(card_label=auc_table("matched ends, right vs wrong", matched, y), alt_label=auc_table("every candidate with delivered ends, right vs the rest", np.ones(len(y), bool), y))
    g = res["B"]["card_label"]["gain S1 -> S2"]
    res["B"]["gate_seed"] = bool(all(x >= 0.05 for x in g["pooled"]) or all(x >= 0.05 for x in g["band"]))
    print("B:", json.dumps(res["B"]["card_label"]["gain S1 -> S2"]), "gate (this seed)", res["B"]["gate_seed"], flush=True)
except ImportError as ex:
    res["B"] = dict(error=str(ex))

# ================================================================ Part C
fused = {}                                   # V6: the fused long-link values of every frame, in time order (the chain)
dms = {v: [None] * len(dump) for v in VARIANTS}
for i in order:
    d = dump[i]
    p = d["link_p"]
    lg = lgs[i]
    mem = MEM.get(i)
    m1 = mem[0] if mem is not None else np.zeros(len(p))
    has = mem is not None
    dms["none"][i] = p >= THR
    dms["V1 average"][i] = ((p + m1) / 2 >= THR) if has else p >= THR
    dms["V2 filter prev>=0.3"][i] = ((p >= THR) & (m1 >= 0.3)) if has else p >= THR
    dms["V3 hold p>=0.25 prev>=0.5"][i] = ((p >= THR) | ((p >= 0.25) & (m1 >= 0.5))) if has else p >= THR
    dms["V4 hold p>=0.1 prev>=0.5"][i] = ((p >= THR) | ((p >= 0.1) & (m1 >= 0.5))) if has else p >= THR
    dms["V5 filter prev>=0.5"][i] = ((p >= THR) & (m1 >= 0.5)) if has else p >= THR
    if has:
        a = prev_of[i]
        m1f = match_prev(i, a, fused[a])[0]
        f = np.where(lg, (p + m1f) / 2, p)
    else:
        f = p.copy()
    fused[i] = f
    dms["V6 recursive average"][i] = f >= THR

ref_card = json.load(open(os.path.join(ROOT, "runs", "e275", f"e273_p_s{SEED}.plain_t0.5", "card.json")))["p"]["graph"]
rec = {v: [] for v in VARIANTS}
t0 = time.time()
for n, (f, d) in enumerate(zip(frames, dump)):
    cached = getattr(f, "_ref", None) is not None
    lg = lgs[n]
    for v in VARIANTS:
        dm = dms[v][n]
        p = d["link_p"]
        p2 = np.where(lg, np.where(dm, np.maximum(p, THR), np.minimum(p, 0.4999)), p)
        nd_ = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"])
        if d.get("s") is not None:
            nd_["s"] = d["s"]
        lk = dict(src=d["link_src"], dst=d["link_dst"], p=p2, long=lg, a=d["link_a"], b=d["link_b"])
        graph, _ = QE.readout(dict(nodes=nd_, links=lk), THR, THR)
        rec[v].append(QE.graph_counts(graph, f))
    if not cached:
        f._lane = f._ref = None
    if (n + 1) % 100 == 0:
        print(f"  frame {n + 1}/{len(dump)}  {time.time() - t0:.0f} s", flush=True)
C = {}
for v in VARIANTS:
    g = QE.graph_numbers(rec[v])
    dl = [delivered(d, dms[v][i]) for i, d in enumerate(dump)]
    pers_v, n_v = persistence(dl)
    nd_v = sum(int((dl[i] & lgs[i]).sum()) for i in range(len(dump)))
    nm_v = sum(int((dl[i] & lgs[i] & INFO[i]["matched"]).sum()) for i in range(len(dump)))
    nr_v = sum(int((dl[i] & lgs[i] & INFO[i]["right"]).sum()) for i in range(len(dump)))
    C[v] = dict({k: float(g[k]) for k in ("topo_F1", "junction_F1", "junction_P", "junction_R", "F1@1.0", "lanes_found", "vert100_dp")}, long_persist=pers_v, persist_n=n_v, delivered_long=nd_v,
                matched_long=nm_v, right_long=nr_v, long_precision_pair=nr_v / max(nm_v, 1))
res["C"] = C
res["repro_none_vs_E275_card"] = {k: C["none"][k] - ref_card[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found")}
res["lane_signature"] = None
try:
    sys.path.insert(0, QE.BP.REPO)
    from autogt import lane_gt
    res["lane_signature"] = lane_gt.signature()
except Exception as ex:                                                  # a record, not a gate
    res["lane_signature"] = f"unknown ({ex})"
json.dump(res, open(os.path.join(OUT, f"p12_{TAG}.json"), "w"), indent=1)
print(json.dumps(dict(A=res["A"], C=res["C"], repro=res["repro_none_vs_E275_card"]), indent=1))

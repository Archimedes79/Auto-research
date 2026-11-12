"""scratch (Mirko's OK of 2026-11-08 'Ja, messen' after oracle_junction.py): how well can the PAIRS (which entry of a junction connects to which exit) be predicted from the GEOMETRY of the ends alone?
Entries = starts, exits = ends of the connector runs of the lane GT (clusters, as oracle_junction.py); candidate pairs = the geometric rule of oracle_junction.py (98 % recall of the true pairs, 4.3 candidates per true pair).
Features of a pair: the turn angle (entry heading -> exit heading, signed, + left) and its size, the chord (length, forward / left offset in the entry's frame and in the exit's frame, the angles between the chord and both headings), the position
of the entry among the parallel entries (similar heading within 12 m: how many lie left / right of it, same for the exit), the number of candidates of the entry and of the exit, the rank of the pair by turn angle and chord length among the
candidates of its entry (and by turn angle among those of its exit), the numbers of entries and exits of the frame. Nothing from the image, the SD map or the network.
Classifier: HistGradientBoosting, trained on every 3rd TRAIN frame, tested on ALL 873 VAL frames; the threshold = the F1 maximum on out-of-fold TRAIN predictions (3 folds of 40-frame blocks).
Reported on VAL: pair AUC, precision / recall at the threshold and at recall 0.9, top-1 hit per entry; and on the 88 frames with a stored E279 output p, the junction map (< 8 m cells) of
  A (true pairs), B (all candidate pairs), C (pairs with probability >= threshold), D (soft: max over pairs of probability x curve heat), against the real target T, next to the E279 output p.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/pair_geometry.py"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
cells = np.stack([CX.ravel(), CY.ravel()], 1)
inbox = (np.abs(cells) <= 47.0).all(1)


def hermite(a, ta, b, tb, k=1.0, step=0.25):
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a + h10 * (k * L * ta) + h01 * b + h11 * (k * L * tb)


def cluster(items, tol_pos=1.5, tol_ang=40.0):
    reps, ids = [], []
    for pos, tan in items:
        for j, (rp, rt) in enumerate(reps):
            if np.linalg.norm(pos - rp) <= tol_pos and np.degrees(np.arccos(np.clip(tan @ rt, -1, 1))) <= tol_ang:
                ids.append(j)
                break
        else:
            reps.append((pos, tan))
            ids.append(len(reps) - 1)
    return reps, ids


def allowed(a, ta, b, tb):
    ch = b - a
    return np.degrees(np.arccos(np.clip(ta @ tb, -1, 1))) <= 130.0 and np.linalg.norm(ch) >= 2.0 and (ch @ ta) > 0 and (ch @ tb) > 0


def sang(u, v):
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def side(P, Tn):
    out = []
    for i in range(len(P)):
        n = np.array([-Tn[i][1], Tn[i][0]])
        d = P - P[i]
        ang = np.degrees(np.arccos(np.clip(Tn @ Tn[i], -1, 1)))
        m = (np.linalg.norm(d, axis=1) <= 12.0) & (ang <= 30.0)
        m[i] = False
        l = d[m] @ n
        out.append((int((l > 1.0).sum()), int((l < -1.0).sum())))
    return out


def analyse(g):
    lane_pts, runs, short = [], [], []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        if (~conn).any():
            lane_pts.append(q[~conn])
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    runs.append((q[k], tan[k], q[j], tan[j]))
                else:
                    short.append(q[k: j + 1])
                k = j + 1
            else:
                k += 1
    if not runs:
        return None
    ent, eid = cluster([(r[0], r[1]) for r in runs])
    ext, xid = cluster([(r[2], r[3]) for r in runs])
    true_pairs = set(zip(eid, xid))
    cand = [(a, b) for a in range(len(ent)) for b in range(len(ext)) if allowed(ent[a][0], ent[a][1], ext[b][0], ext[b][1])]
    if not cand:
        return None
    y = np.array([(a, b) in true_pairs for a, b in cand], int)
    E, TE = np.array([e[0] for e in ent]), np.array([e[1] for e in ent])
    X_, TX = np.array([x[0] for x in ext]), np.array([x[1] for x in ext])
    sa, sb = side(E, TE), side(X_, TX)
    na, nb = Counter(a for a, b in cand), Counter(b for a, b in cand)
    turn = np.array([sang(ent[a][1], ext[b][1]) for a, b in cand])
    L = np.array([np.linalg.norm(ext[b][0] - ent[a][0]) for a, b in cand])
    rk_turn_a, rk_len_a, rk_turn_b = np.zeros(len(cand)), np.zeros(len(cand)), np.zeros(len(cand))
    for a in set(x[0] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if aa == a]
        for arr, vals in ((rk_turn_a, np.abs(turn[idx])), (rk_len_a, L[idx])):
            r = np.argsort(np.argsort(vals))
            arr[idx] = r / max(len(idx) - 1, 1)
    for b in set(x[1] for x in cand):
        idx = [i for i, (aa, bb) in enumerate(cand) if bb == b]
        r = np.argsort(np.argsort(np.abs(turn[idx])))
        rk_turn_b[idx] = r / max(len(idx) - 1, 1)
    F = []
    for i, (a, b) in enumerate(cand):
        pa, ta, pb, tb = ent[a][0], ent[a][1], ext[b][0], ext[b][1]
        ch = pb - pa
        na_ = np.array([-ta[1], ta[0]])
        nb_ = np.array([-tb[1], tb[0]])
        F.append([turn[i], abs(turn[i]), L[i], ch @ ta, ch @ na_, ch @ tb, ch @ nb_, sang(ta, ch), sang(tb, ch), sa[a][0], sa[a][1], sb[b][0], sb[b][1], na[a], nb[b], rk_turn_a[i], rk_len_a[i], rk_turn_b[i], len(ent), len(ext)])
    return dict(base=lane_pts + short, ent=ent, ext=ext, cand=cand, y=y, X=np.array(F, float), n_true=len(true_pairs))


# ---- TRAIN set
ds = LD.QData("train", map_mode="exact", pixels=False)
ks_tr = list(ds.ks)[::3]
toks_tr = [ds.tokens[int(ds.bd.rows[k])] for k in ks_tr]
POSE = {part: np.load(os.path.expanduser(f"~/exp_gcar_b123/poses_{part}.npz")) for part in ("train", "val")}
seen_db = {}                                                       # map location -> list of (global entry position, global heading) of the TRAIN junctions


def entries_global(r, part, row):
    Rr, tt = POSE[part]["R"][row], POSE[part]["t"][row]
    return np.array([e[0] for e in r["ent"]]) @ Rr.T + tt, np.array([e[1] for e in r["ent"]]) @ Rr.T


Xtr, ytr, gtr = [], [], []
for i, tk in enumerate(toks_tr):
    r = analyse(ds.lg.load(tk))
    if r is None:
        continue
    row_i = int(ds.bd.rows[ks_tr[i]])
    gp, gh = entries_global(r, "train", row_i)
    seen_db.setdefault(int(POSE["train"]["loc"][row_i]), []).append((gp, gh))
    Xtr.append(r["X"])
    ytr.append(r["y"])
    gtr.append(np.full(len(r["y"]), i // 40))
    if (i + 1) % 500 == 0:
        print(i + 1, "train frames", flush=True)
Xtr, ytr, gtr = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(gtr)
print("TRAIN candidates", len(ytr), "true", int(ytr.sum()), flush=True)
mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0)  # noqa: E731
oof = np.zeros(len(ytr))
folds = gtr % 3
for fo in range(3):
    m = mk().fit(Xtr[folds != fo], ytr[folds != fo])
    oof[folds == fo] = m.predict_proba(Xtr[folds == fo])[:, 1]
best = (0, 0.5)
for t in np.linspace(0.05, 0.95, 91):
    pr = oof >= t
    tp = (pr & (ytr == 1)).sum()
    f1 = 2 * tp / max(pr.sum() + ytr.sum(), 1)
    best = max(best, (f1, float(t)))
tau = best[1]
print(f"out-of-fold TRAIN: AUC {roc_auc_score(ytr, oof):.3f}  threshold {tau:.2f} (F1 {best[0]:.3f})", flush=True)
model = mk().fit(Xtr, ytr)

# ---- VAL: all 873 frames
frames = QE.val_frames()
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
toks88 = [str(t) for t in z["tokens"]][:88]
P0, T0, DJ0, DG0 = (z[k].astype(np.float32)[:88] for k in ("p", "t", "dj", "dg"))
Xv, yv, per_frame = [], [], {}
for i, f in enumerate(frames):
    r = analyse(f.lane())
    f._lane = f._ref = None
    if r is None:
        continue
    r["prob"] = model.predict_proba(r["X"])[:, 1]
    gp, gh = entries_global(r, "val", f.row)
    db = seen_db.get(int(POSE["val"]["loc"][f.row]), [])
    if db:
        DP = np.concatenate([d[0] for d in db])
        DH = np.concatenate([d[1] for d in db])
        tree_db = cKDTree(DP)
        hits = []
        for p_, h_ in zip(gp, gh):
            nn = tree_db.query_ball_point(p_, 5.0)
            hits.append(bool(nn) and bool((DH[nn] @ h_ >= np.cos(np.radians(30.0))).any()))
        r["seen_frac"] = float(np.mean(hits))
    else:
        r["seen_frac"] = 0.0
    per_frame[f.token] = r
    Xv.append(r["X"])
    yv.append(r["y"])
    if (i + 1) % 200 == 0:
        print(i + 1, "val frames", flush=True)
yv = np.concatenate(yv)
pv = np.concatenate([r["prob"] for r in per_frame.values()])
res = dict(train_oof_auc=float(roc_auc_score(ytr, oof)), tau=tau, train_candidates=int(len(ytr)), train_precision_all=float(ytr.mean()))
res["val_pairs"] = dict(candidates=int(len(yv)), true=int(yv.sum()), precision_all_pairs=float(yv.mean()), auc=float(roc_auc_score(yv, pv)))
pr = pv >= tau
res["val_pairs"].update(precision_at_tau=float((pr & (yv == 1)).sum() / max(pr.sum(), 1)), recall_at_tau=float((pr & (yv == 1)).sum() / max(yv.sum(), 1)), kept_per_true=float(pr.sum() / max(yv.sum(), 1)))
for target in (0.9, 0.95):
    for t in np.linspace(0.95, 0.01, 95):
        prr = pv >= t
        rec = (prr & (yv == 1)).sum() / yv.sum()
        if rec >= target:
            res["val_pairs"][f"precision_at_recall_{target}"] = float((prr & (yv == 1)).sum() / prr.sum())
            res["val_pairs"][f"threshold_at_recall_{target}"] = float(t)
            break
hit, n_ent = 0, 0
for r in per_frame.values():
    for a in set(x[0] for x in r["cand"]):
        idx = [i for i, (aa, bb) in enumerate(r["cand"]) if aa == a]
        if r["y"][idx].sum() == 0:
            continue
        n_ent += 1
        hit += int(r["y"][idx][int(np.argmax(r["prob"][idx]))] == 1)
res["val_pairs"]["top1_hit_per_entry"] = hit / max(n_ent, 1)
print("VAL pairs:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res["val_pairs"].items()}), flush=True)
# seen / unseen junctions: a VAL entry counts as seen when a TRAIN entry of the same map location lies within 5 m with a heading within 30 deg
res["seen_split"] = {}
for name, lo, hi in (("unseen (seen share <= 0.2)", -1, 0.2), ("partly seen", 0.2, 0.8), ("seen (share >= 0.8)", 0.8, 2)):
    sel = [r for r in per_frame.values() if lo < r["seen_frac"] <= hi] if lo >= 0 or name.startswith("unseen") else []
    if name.startswith("partly"):
        sel = [r for r in per_frame.values() if 0.2 < r["seen_frac"] < 0.8]
    if name.startswith("seen"):
        sel = [r for r in per_frame.values() if r["seen_frac"] >= 0.8]
    if name.startswith("unseen"):
        sel = [r for r in per_frame.values() if r["seen_frac"] <= 0.2]
    if not sel:
        continue
    yy = np.concatenate([r["y"] for r in sel])
    pp = np.concatenate([r["prob"] for r in sel])
    ok = 0 < yy.sum() < len(yy)
    prr = pp >= tau
    res["seen_split"][name] = dict(frames=len(sel), candidates=int(len(yy)), true=int(yy.sum()), auc=float(roc_auc_score(yy, pp)) if ok else None, precision_all=float(yy.mean()),
                                   precision_at_tau=float((prr & (yy == 1)).sum() / max(prr.sum(), 1)), recall_at_tau=float((prr & (yy == 1)).sum() / max(yy.sum(), 1)))
    print(" ", name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res["seen_split"][name].items()}), flush=True)


# ---- the junction map on the 88 frames
def heat_pts(pts):
    pts = np.concatenate(pts) if isinstance(pts, list) else pts
    pts = pts[(np.abs(pts) < 48.0).all(1)]
    return np.exp(-cKDTree(pts).query(cells)[0] ** 2 / 2.0).astype(np.float32)


maps = {k: [] for k in ("A true pairs", "B all pairs", "C predicted pairs", "D soft (prob x curve)")}
used = []
for i, tk in enumerate(toks88):
    r = per_frame.get(tk)
    if r is None:
        used.append(False)
        continue
    used.append(True)
    curves = [hermite(r["ent"][a][0], r["ent"][a][1], r["ext"][b][0], r["ext"][b][1]) for a, b in r["cand"]]
    base = r["base"]
    base_h = heat_pts(base) if base else np.zeros(len(cells), np.float32)
    ch = [heat_pts(cv) for cv in curves]
    sel_true = [k for k in range(len(curves)) if r["y"][k]]
    sel_pred = [k for k in range(len(curves)) if r["prob"][k] >= tau]
    maps["A true pairs"].append(np.maximum.reduce([base_h] + [ch[k] for k in sel_true]))
    maps["B all pairs"].append(np.maximum.reduce([base_h] + ch))
    maps["C predicted pairs"].append(np.maximum.reduce([base_h] + [ch[k] for k in sel_pred]))
    maps["D soft (prob x curve)"].append(np.maximum.reduce([base_h] + [r["prob"][k] * ch[k] for k in range(len(curves))]))
rows = [i for i, u in enumerate(used) if u]
Tn, DJn, DGn, Pn = T0[rows], DJ0[rows], DG0[rows], P0[rows]


def measure(X, sel=None):
    a = dict(n=0, sx=0.0, st=0.0, sxx=0.0, stt=0.0, sxt=0.0, ab=0.0, hit=0, onx=0, ont=0, gap=0, gap_hi=0)
    for i in (range(len(X)) if sel is None else sel):
        m = (DJn[i] < 8.0) & inbox & ((X[i] > 0.05) | (Tn[i] > 0.05))
        mg = (DJn[i] < 8.0) & inbox & (Tn[i] < 0.1) & (DGn[i] < 6.0)
        mi = (DJn[i] < 8.0) & inbox
        a["n"] += int(m.sum())
        a["sx"] += float(X[i][m].sum())
        a["st"] += float(Tn[i][m].sum())
        a["sxx"] += float((X[i][m] ** 2).sum())
        a["stt"] += float((Tn[i][m] ** 2).sum())
        a["sxt"] += float((X[i][m] * Tn[i][m]).sum())
        a["ab"] += float(np.abs(X[i][m] - Tn[i][m]).sum())
        ox, ot = (X[i] >= 0.5) & mi, (Tn[i] >= 0.5) & mi
        a["hit"] += int((ox & ot).sum())
        a["onx"] += int(ox.sum())
        a["ont"] += int(ot.sum())
        a["gap"] += int(mg.sum())
        a["gap_hi"] += int((X[i][mg] > 0.3).sum())
    n = max(a["n"], 1)
    cov = a["sxt"] / n - a["sx"] / n * a["st"] / n
    vx, vt = a["sxx"] / n - (a["sx"] / n) ** 2, a["stt"] / n - (a["st"] / n) ** 2
    return dict(corr=float(cov / np.sqrt(max(vx * vt, 1e-12))), mean_abs=a["ab"] / n, iou=a["hit"] / max(a["onx"] + a["ont"] - a["hit"], 1), gap_fill=a["gap_hi"] / max(a["gap"], 1))


res["map"] = {k: measure(np.stack(v)) for k, v in maps.items()}
res["map"]["E279 output p"] = measure(Pn)
res["map_frames"] = len(rows)
sf = [per_frame[toks88[i]]["seen_frac"] for i in rows]
unseen = [j for j, s in enumerate(sf) if s <= 0.2]
res["map_unseen"] = {k: measure(np.stack(v), unseen) for k, v in maps.items()} if len(unseen) >= 8 else {}
if res["map_unseen"]:
    res["map_unseen"]["E279 output p"] = measure(Pn, unseen)
res["map_unseen_frames"] = len(unseen)
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "pair_geometry.json"), "w"), indent=1)
print(f"junction cells (< 8 m), {len(rows)} VAL frames, against the real target T")
for k, v in res["map"].items():
    print(f"  {k:24s} corr {v['corr']:.3f}  mean |X - T| {v['mean_abs']:.3f}  IoU(0.5) {100 * v['iou']:5.1f} %  empty cells filled (> 0.3) {100 * v['gap_fill']:5.1f} %")
print(f"only the {len(unseen)} frames whose junction is unseen in TRAIN")
for k, v in res["map_unseen"].items():
    print(f"  {k:24s} corr {v['corr']:.3f}  mean |X - T| {v['mean_abs']:.3f}  IoU(0.5) {100 * v['iou']:5.1f} %  empty cells filled (> 0.3) {100 * v['gap_fill']:5.1f} %")

"""E384 referee probe (CPU, VAL only, B cache seed S): does the previous keyframe output of B carry information about THIS frame node existence beyond the current logit?
(1) coverage ceiling: GT nodes with no valid token at t; share with a delivered (p>=0.5) warped past token near; nulls identity / random frame; carry-over precision of dropped past tokens.
(2) existence ranking: GBM on [e] vs [e + past features] (pose / identity / random-frame null), 5-fold CV grouped by scene, AP over valid tokens, scene bootstrap of the AP difference.
usage: python e384_step0.py <seed> <out.json>"""
import json
import os
import sys
import time

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import lane_past_data as LPD  # noqa: E402
import lane_tok_data as LT  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

seed = int(sys.argv[1])
out = sys.argv[2]
d = LP.cache_dir("val", seed, "e292_a")
ext = LPD.load_ext(d)
c = LC.TokCache(d)
ds = LT.TokData("val", res=2.0, feats=False)
assert ds.tokens == c.tokens and len(c) == 873
LPD.check_rows("val", ds.rows, expect=None)
scene = LPD.scene_ids(ext["prev"])
rng = np.random.default_rng(0)
RAD, COS = 1.5, 0.5


def arr(i, k):
    return np.array(c.f[k][i])


def near(P, D, A, B):
    if len(P) == 0 or len(A) == 0:
        return np.zeros((len(P), len(A)), bool)
    return (np.linalg.norm(P[:, None] - A[None], axis=2) <= RAD) & (D @ B.T >= COS)


def pfeat(P, D, A, B, pa):
    """past features of tokens P/D from past tokens A/B with probabilities pa: pmax within near, count p>=0.5 within near, nearest delivered distance (clipped 8)"""
    n = len(P)
    if len(A) == 0:
        return np.zeros((n, 3), np.float32) + np.array([0, 0, 8.0], np.float32)
    nr = near(P, D, A, B)
    pmax = (nr * pa[None]).max(1)
    n50 = (nr & (pa[None] >= 0.5)).sum(1)
    dist = np.linalg.norm(P[:, None] - A[None], axis=2)
    dist = np.where((D @ B.T >= COS) & (pa[None] >= 0.5), dist, 99.0).min(1)
    return np.stack([pmax, n50, np.minimum(dist, 8.0)], 1).astype(np.float32)


acc = {}


def add(k, v):
    acc[k] = acc.get(k, 0) + int(v)


X0, XP, XI, XR, Y, G, FR = [], [], [], [], [], [], []
t0 = time.time()
for i in range(len(c)):
    if ext["prev"][i] < 0:
        continue
    ip = int(ext["prev"][i])
    v = arr(i, "valid")
    pos, dirv = arr(i, "pos").astype(float)[v], arr(i, "dirv").astype(float)[v]
    e = arr(i, "e").astype(float)[v]
    gid = arr(i, "gt_id")[v]
    pv = arr(ip, "valid")
    pp_all, pd_all, pe_all = arr(ip, "pos").astype(float)[pv], arr(ip, "dirv").astype(float)[pv], arr(ip, "e_p").astype(float)[pv]
    Rm, tv = (a.astype(float) for a in LPD.pose_of(ext, i))
    Pw, Dw = pp_all @ Rm.T + tv, pd_all @ Rm.T
    j = int(rng.integers(len(c)))
    jv = arr(j, "valid")
    Pr, Dr, pr = arr(j, "pos").astype(float)[jv], arr(j, "dirv").astype(float)[jv], arr(j, "e_p").astype(float)[jv]
    X0.append(e[:, None].astype(np.float32))
    XP.append(pfeat(pos, dirv, Pw, Dw, pe_all))
    XI.append(pfeat(pos, dirv, pp_all, pd_all, pe_all))
    XR.append(pfeat(pos, dirv, Pr, Dr, pr))
    Y.append(gid >= 0)
    G.append(np.full(len(e), scene[i]))
    FR.append(np.full(len(e), i))
    t = ds.table(i)
    gpos, gdir = np.asarray(t["pos"], float), np.asarray(t["dirv"], float)
    M = len(gpos)
    if M == 0:
        continue
    cov = near(pos, dirv, gpos, gdir).any(0)
    dl = pe_all >= 0.5
    drr = pr >= 0.5
    sup = {"pose": near(Pw[dl], Dw[dl], gpos, gdir).any(0) if dl.any() else np.zeros(M, bool),
           "identity": near(pp_all[dl], pd_all[dl], gpos, gdir).any(0) if dl.any() else np.zeros(M, bool),
           "random": near(Pr[drr], Dr[drr], gpos, gdir).any(0) if drr.any() else np.zeros(M, bool)}
    add("gt", M)
    add("uncov", (~cov).sum())
    add("cov", cov.sum())
    for k, s in sup.items():
        add(f"uncov_sup_{k}", (s & ~cov).sum())
        add(f"cov_sup_{k}", (s & cov).sum())
    for k, (A, B) in {"pose": (Pw[dl], Dw[dl]), "identity": (pp_all[dl], pd_all[dl]), "random": (Pr[drr], Dr[drr])}.items():
        if len(A) == 0:
            continue
        now = near(A, B, pos, dirv).any(1)
        drop = ~now
        tr = near(A, B, gpos, gdir).any(1)
        add(f"drop_{k}", drop.sum())
        add(f"drop_true_{k}", (drop & tr).sum())
        add(f"keep_{k}", now.sum())
print("collected", len(Y), "frames", round(time.time() - t0), "s", flush=True)
X0, XP, XI, XR, Y, G, FR = (np.concatenate(a) for a in (X0, XP, XI, XR, Y, G, FR))
nfr = len(set(FR.tolist()))


def r(a, b):
    return float(acc.get(a, 0)) / max(float(acc.get(b, 0)), 1.0)


res = dict(seed=seed, frames_with_past=nfr, gt=acc["gt"], uncovered_share=r("uncov", "gt"),
           uncovered_supported={k: r(f"uncov_sup_{k}", "uncov") for k in ("pose", "identity", "random")},
           covered_supported={k: r(f"cov_sup_{k}", "cov") for k in ("pose", "identity", "random")},
           coverage_ceiling_pts={k: r(f"uncov_sup_{k}", "gt") for k in ("pose", "identity", "random")},
           dropped_past_tokens={k: dict(n=acc.get(f"drop_{k}", 0), true=r(f"drop_true_{k}", f"drop_{k}"), per_frame=acc.get(f"drop_{k}", 0) / nfr) for k in ("pose", "identity", "random")},
           tokens=int(len(Y)), assigned_share=float(Y.mean()))
print(json.dumps(res, indent=1), flush=True)
models = {"base": X0, "pose": np.hstack([X0, XP]), "identity": np.hstack([X0, XI]), "random": np.hstack([X0, XR])}
oof = {k: np.zeros(len(Y)) for k in models}
for tr_i, te_i in GroupKFold(5).split(X0, Y, G):
    for k, X in models.items():
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
        m.fit(X[tr_i], Y[tr_i])
        oof[k][te_i] = m.predict_proba(X[te_i])[:, 1]
    print("fold done", round(time.time() - t0), "s", flush=True)
ap = {k: float(average_precision_score(Y, oof[k])) for k in models}
ap["raw_p(e)"] = float(average_precision_score(Y, X0[:, 0]))
res["ap"] = ap
sc = np.unique(G)
idx = {s: np.where(G == s)[0] for s in sc}
bd = {k: [] for k in ("pose", "identity", "random", "pose_minus_identity")}
for b in range(100):
    ii = np.concatenate([idx[s] for s in rng.choice(sc, len(sc))])
    a = {k: average_precision_score(Y[ii], oof[k][ii]) for k in models}
    for k in ("pose", "identity", "random"):
        bd[k].append(a[k] - a["base"])
    bd["pose_minus_identity"].append(a["pose"] - a["identity"])
res["ap_delta_vs_base"] = {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))) for k, v in bd.items()}
res["ap_point_delta"] = {k: ap[k] - ap["base"] for k in ("pose", "identity", "random")}
pbase = float(Y[X0[:, 0] >= 0].mean())


def recall_at_prec(s, y, p0):
    pr_, rc_, _ = precision_recall_curve(y, s)
    ok = pr_ >= p0
    return float(rc_[ok].max()) if ok.any() else 0.0


res["precision_at_p50_base"] = pbase
res["recall_at_that_precision"] = {k: recall_at_prec(oof[k], Y, pbase) for k in models}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))

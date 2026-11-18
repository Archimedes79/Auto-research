"""E384 referee probe b (CPU, VAL): the coverage ceiling when the past support is ANY valid token (p>0.05) / p>=0.2 / p>=0.5 of t-1, warped; nulls identity, random frame; carry-over precision of the dropped ones
usage: python e384_step0b.py <seed> <out.json>"""
import json
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402
import lane_past_data as LPD  # noqa: E402
import lane_tok_data as LT  # noqa: E402

seed = int(sys.argv[1])
out = sys.argv[2]
d = LP.cache_dir("val", seed, "e292_a")
ext = LPD.load_ext(d)
c = LC.TokCache(d)
ds = LT.TokData("val", res=2.0, feats=False)
assert ds.tokens == c.tokens and len(c) == 873
LPD.check_rows("val", ds.rows, expect=None)
rng = np.random.default_rng(0)
RAD, COS = 1.5, 0.5
TH = {"p>=0.5": 0.5, "p>=0.2": 0.2, "valid(p>0.05)": 0.0}


def arr(i, k):
    return np.array(c.f[k][i])


def near(P, D, A, B):
    if len(P) == 0 or len(A) == 0:
        return np.zeros((len(P), len(A)), bool)
    return (np.linalg.norm(P[:, None] - A[None], axis=2) <= RAD) & (D @ B.T >= COS)


acc = {}


def add(k, v):
    acc[k] = acc.get(k, 0) + int(v)


nfr = 0
for i in range(len(c)):
    if ext["prev"][i] < 0:
        continue
    ip = int(ext["prev"][i])
    v = arr(i, "valid")
    pos, dirv = arr(i, "pos").astype(float)[v], arr(i, "dirv").astype(float)[v]
    pv = arr(ip, "valid")
    pp, pd, pe = arr(ip, "pos").astype(float)[pv], arr(ip, "dirv").astype(float)[pv], arr(ip, "e_p").astype(float)[pv]
    Rm, tv = (a.astype(float) for a in LPD.pose_of(ext, i))
    Pw, Dw = pp @ Rm.T + tv, pd @ Rm.T
    j = int(rng.integers(len(c)))
    jv = arr(j, "valid")
    Pr, Dr, pr = arr(j, "pos").astype(float)[jv], arr(j, "dirv").astype(float)[jv], arr(j, "e_p").astype(float)[jv]
    t = ds.table(i)
    gpos, gdir = np.asarray(t["pos"], float), np.asarray(t["dirv"], float)
    M = len(gpos)
    if M == 0:
        continue
    nfr += 1
    cov = near(pos, dirv, gpos, gdir).any(0)
    flags = np.asarray(t["flags"]).astype(bool)
    topo = flags[:, :6].any(1)
    add("gt", M)
    add("uncov", (~cov).sum())
    add("uncov_topo", ((~cov) & topo).sum())
    add("gt_topo", topo.sum())
    for name, th in TH.items():
        for k, (A, B, p) in {"pose": (Pw, Dw, pe), "identity": (pp, pd, pe), "random": (Pr, Dr, pr)}.items():
            sel = p >= th if th > 0 else np.ones(len(p), bool)
            A_, B_ = A[sel], B[sel]
            s = near(A_, B_, gpos, gdir).any(0) if len(A_) else np.zeros(M, bool)
            add(f"{name}|{k}|uncov_sup", (s & ~cov).sum())
            add(f"{name}|{k}|uncov_sup_topo", (s & ~cov & topo).sum())
            if len(A_):
                now = near(A_, B_, pos, dirv).any(1)
                tr = near(A_, B_, gpos, gdir).any(1)
                add(f"{name}|{k}|drop", (~now).sum())
                add(f"{name}|{k}|drop_true", ((~now) & tr).sum())
res = dict(seed=seed, frames=nfr, gt=acc["gt"], uncovered=acc["uncov"] / acc["gt"], uncovered_topo_share_of_gt=acc["uncov_topo"] / acc["gt"], topo_share_of_gt=acc["gt_topo"] / acc["gt"])
for name in TH:
    for k in ("pose", "identity", "random"):
        res[f"{name}|{k}"] = dict(ceiling_pts=acc[f"{name}|{k}|uncov_sup"] / acc["gt"], ceiling_topo_pts=acc[f"{name}|{k}|uncov_sup_topo"] / acc["gt"],
                                  dropped_per_frame=acc.get(f"{name}|{k}|drop", 0) / nfr, dropped_true=acc.get(f"{name}|{k}|drop_true", 0) / max(acc.get(f"{name}|{k}|drop", 0), 1))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))

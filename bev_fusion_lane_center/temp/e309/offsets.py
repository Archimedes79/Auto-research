"""E309 scratch (2026-11-12 evening): is the position error of the delivered lines a smooth offset (a whole link sits 1 - 2 m beside the GT: smoothness cannot fix it) or a jitter along the line (a smoothness constraint would help)?
For every delivered line (0.5 m steps) the signed lateral offset to the nearest GT point (along the GT normal) of its points within 3 m of the GT: the mean per line (the offset), the std around a straight-line fit per line (the jitter), and where the lines are (distance from the ego).
usage: wsl bash tools/env/_py14.sh e309/offsets.py [seed]   (CPU, one thread)"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", f"perframe_s{seed}.pkl"), "rb"))
DS = 0.5
BOX = 48.0


def resample(p, ds=DS):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


out = {}
RES = {}
for k in ("B", "e312"):
    mean_off, jit, rng, nlen = [], [], [], []
    tot_m2 = tot_j2 = 0.0
    for rec in PF:
        G = [resample(p) for p in rec["gt_pieces"] if len(p) > 2]
        if not G:
            continue
        GP = np.vstack(G)
        GT = np.vstack([np.gradient(q, axis=0) / DS if len(q) > 2 else np.zeros_like(q) for q in G])
        n = np.linalg.norm(GT, axis=1, keepdims=True)
        GT = GT / np.maximum(n, 1e-6)
        tree = cKDTree(GP)
        for l in rec[k]["lines"]:
            q = resample(l)
            if len(q) < 10 or (np.abs(q) > BOX).any():
                continue
            d, i = tree.query(q)
            m = d < 3.0
            if m.sum() < 10:
                continue
            nrm = np.c_[-GT[i, 1], GT[i, 0]]
            s = np.einsum("ij,ij->i", q - GP[i], nrm)[m]
            t = np.arange(len(s)) * DS
            A = np.c_[np.ones_like(t), t]
            coef, *_ = np.linalg.lstsq(A, s, rcond=None)
            res = s - A @ coef
            mean_off.append(float(coef[0] + coef[1] * t.mean()))
            jit.append(float(res.std()))
            rng.append(float(np.linalg.norm(q[m].mean(0))))
            nlen.append(len(s) * DS)
            tot_m2 += float(((coef[0] + coef[1] * t.mean()) ** 2) * len(s))
            tot_j2 += float((res ** 2).sum())
    mo, jt, rg = np.abs(mean_off), np.array(jit), np.array(rng)
    print(f"{k}: {len(mo)} lines within 3 m of the GT; |mean lateral offset| median {np.median(mo):.2f} m p90 {np.percentile(mo, 90):.2f} m;"
          f" jitter around the fitted line (std) median {np.median(jt):.2f} m p90 {np.percentile(jt, 90):.2f} m;"
          f" share of the squared lateral error that is the per-line offset (the rest: slope + jitter) {100 * tot_m2 / (tot_m2 + tot_j2):.0f} %")
    RES[k] = dict(n=len(mo), off_med=float(np.median(mo)), off_p90=float(np.percentile(mo, 90)), jit_med=float(np.median(jt)), jit_p90=float(np.percentile(jt, 90)), off_share=float(100 * tot_m2 / (tot_m2 + tot_j2)), by_range={})
    for lo, hi in ((0, 15), (15, 30), (30, 48)):
        RES[k]["by_range"][f"{lo}-{hi}"] = dict(n=int(((rg >= lo) & (rg < hi)).sum()), off_med=float(np.median(mo[(rg >= lo) & (rg < hi)])), jit_med=float(np.median(jt[(rg >= lo) & (rg < hi)])))
        w = (rg >= lo) & (rg < hi)
        print(f"   range {lo:2d}-{hi:2d} m: lines {int(w.sum()):5d}  |offset| median {np.median(mo[w]):.2f}  jitter median {np.median(jt[w]):.2f}")

import json
json.dump(RES, open(os.path.join(ROOT, "temp", "e309", f"offsets_s{seed}.json"), "w"), indent=1)

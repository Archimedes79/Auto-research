"""E309 scratch (2026-11-12 evening): how many lines / how much length each variant delivers, how much of it is wrong (1 m / 2 m from the l6 GT), and how smooth it is (heading change per step, kinks) against the GT.
usage: wsl bash tools/env/_py14.sh e309/lines_stats.py [seed]  (reads temp/e309/perframe_s<seed>.pkl; CPU, one thread)"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", f"perframe_s{seed}.pkl"), "rb"))
BOX = 48.0
VARS = ("B", "e284", "e287p2", "e312")
DS = 0.5


def resample(p, ds=DS):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


def inbox(A):
    return A[(np.abs(A[:, 0]) < BOX) & (np.abs(A[:, 1]) < BOX)] if len(A) else A


def headings(q):
    d = np.diff(q, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def turn(q):
    """heading change (deg) between consecutive 0.5 m steps of one resampled line"""
    if len(q) < 3:
        return np.zeros(0)
    h = headings(q)
    dh = np.angle(np.exp(1j * np.diff(h)))
    return np.degrees(np.abs(dh))


def joins(lines, tol=0.6):
    """heading mismatch (deg) at the places where one line ends and another starts (end of A within tol of the start of B): the node joins"""
    ends = [(l[-1], headings(l[-6:])[-1] if len(l) > 5 else None) for l in lines if len(l) > 5]
    starts = [(l[0], headings(l[:6])[0]) for l in lines if len(l) > 5]
    if not ends or not starts:
        return []
    sp = np.array([s[0] for s in starts])
    t = cKDTree(sp)
    out = []
    for p, h in ends:
        for j in t.query_ball_point(p, tol):
            out.append(abs(np.degrees(np.angle(np.exp(1j * (starts[j][1] - h))))))
    return out


rows = {k: dict(n_lines=[], length=[], bad1=0.0, bad2=0.0, tot=0.0, turn=[], joins=[]) for k in VARS + ("GT",)}
extra = dict(len=0.0, bad1=0.0, bad2=0.0, n=0)
for rec in PF:
    G = [resample(p) for p in rec["gt_pieces"] if len(p) > 1]
    GA = inbox(np.vstack(G)) if G else np.zeros((0, 2))
    tg = cKDTree(GA) if len(GA) else None
    r = rows["GT"]
    r["n_lines"].append(len(G))
    r["length"].append(len(GA) * DS)
    for q in G:
        r["turn"].append(turn(q))
    r["joins"] += joins(G)
    PA = {}
    for k in VARS:
        L = [resample(l) for l in rec[k]["lines"] if len(l) > 1]
        P = inbox(np.vstack(L)) if L else np.zeros((0, 2))
        PA[k] = P
        r = rows[k]
        r["n_lines"].append(len(L))
        r["length"].append(len(P) * DS)
        if len(P) and tg is not None:
            d = tg.query(P)[0]
            r["bad1"] += float((d > 1.0).sum()) * DS
            r["bad2"] += float((d > 2.0).sum()) * DS
        r["tot"] += len(P) * DS
        for q in L:
            r["turn"].append(turn(q))
        r["joins"] += joins(L)
    # the extra length of E312 over B: points of E312 that are more than 1 m from every B point
    if len(PA["e312"]) and len(PA["B"]):
        d_b = cKDTree(PA["B"]).query(PA["e312"])[0]
        m = d_b > 1.0
        if tg is not None and m.any():
            d = tg.query(PA["e312"][m])[0]
            extra["len"] += float(m.sum()) * DS
            extra["bad1"] += float((d > 1.0).sum()) * DS
            extra["bad2"] += float((d > 2.0).sum()) * DS
    extra["n"] += 1

RES = {}
print(f"seed {seed}, {len(PF)} frames, box +-{BOX} m, resample {DS} m")
print(f"{'variant':8s} {'lines/frame':>11s} {'len m/frame':>11s} {'>1m %':>7s} {'>2m %':>7s} {'turn>10deg per 100m':>20s} {'turn>25deg per 100m':>20s} {'joins n/frame':>14s} {'join kink med/p90 deg':>22s}")
for k in ("GT",) + VARS:
    r = rows[k]
    T = np.concatenate(r["turn"]) if r["turn"] else np.zeros(0)
    totlen = float(np.sum(r["length"]))
    j = np.array(r["joins"]) if r["joins"] else np.zeros(1)
    bad1 = 100 * r["bad1"] / max(r["tot"], 1e-9) if k != "GT" else float("nan")
    bad2 = 100 * r["bad2"] / max(r["tot"], 1e-9) if k != "GT" else float("nan")
    RES[k] = dict(lines_per_frame=float(np.mean(r["n_lines"])), length_m=float(np.mean(r["length"])), bad1=bad1, bad2=bad2, turn10=float(100 * (T > 10).sum() * DS / max(totlen, 1e-9)), turn25=float(100 * (T > 25).sum() * DS / max(totlen, 1e-9)), joins_per_frame=len(r["joins"]) / len(PF), join_med=float(np.median(j)), join_p90=float(np.percentile(j, 90)))
    print(f"{k:8s} {np.mean(r['n_lines']):11.1f} {np.mean(r['length']):11.1f} {bad1:7.2f} {bad2:7.2f} {100 * (T > 10).sum() * DS / max(totlen, 1e-9):20.2f} {100 * (T > 25).sum() * DS / max(totlen, 1e-9):20.2f} {len(r['joins']) / len(PF):14.1f} {np.median(j):10.1f} /{np.percentile(j, 90):6.1f}")
print(f"E312 length that B does not have (> 1 m from every B point): {extra['len'] / extra['n']:.1f} m/frame; of it > 1 m from GT {100 * extra['bad1'] / max(extra['len'], 1e-9):.1f} %, > 2 m from GT {100 * extra['bad2'] / max(extra['len'], 1e-9):.1f} %")

RES["e312_extra_over_B"] = dict(length_m=extra["len"] / extra["n"], bad1=100 * extra["bad1"] / max(extra["len"], 1e-9), bad2=100 * extra["bad2"] / max(extra["len"], 1e-9))
import json
json.dump(RES, open(os.path.join(ROOT, "temp", "e309", f"lines_stats_s{seed}.json"), "w"), indent=1)

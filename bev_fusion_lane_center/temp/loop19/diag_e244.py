"""E244 layer-1 diagnostic (CPU): the gradient dead zone of the recall term on the cached j1 / j1eq / j1rc raw nodes of VAL 873.

    wsl bash tools/env/_py14.sh temp/loop19/diag_e244.py

Replicates the soft minimum of training/losses/p16_loss.py graph_losses on the cached nodes: for every 2nd ground-truth lane point (1 m spacing,
|x|,|y| < 48) dg = min over valid edges (parent -> node) of [distance to the segment + 2 m x (1 - pi)], the term is clamped at 3 m (dead zone: dg >= 3
gives no gradient). Per range bin (< 25 m, 25-40 m, > 40 m) and per model:
  dead      share of points with dg >= 3 m (no gradient today)
  rescued   share with 3 <= dg < 8 m (a clamp of 8 m would give them a gradient)
  hopeless  share with dg >= 8 m (no edge within reach even then)
  covered   share with a valid edge within 1 m (geometry only); dead_covered = covered points that are dead only because pi is small (price)
  missed far point statistics: of the points NOT covered (geometric distance >= 1 m), the share that sits in the dead zone
The kill criterion of the card: < 15 % of the missed far points in the dead zone and arm b not planned -> mechanism 1 wrong.
Nodes of the cache are the inference nodes (valid = the builder's own validity at tau-free level), pi = path validity; this is the same tensor the loss sees except
that training nodes come from a train-mode forward pass (teacher forcing share); read it as a proxy, not as the training distribution.
"""
import os
import pickle

import numpy as np

import p16_data as PD

HOME = os.path.expanduser("~")
RAW = {"j1": f"{HOME}/exp_gcar_b123/p18/raw_val873_e123_D7_r3_j1_192.pkl", "j1eq": f"{HOME}/exp_gcar_b123/p18/raw_val873_e18_D7_j1eq_192.pkl",
       "j1rc": f"{HOME}/exp_gcar_b123/p18/raw_val873_e18_D7_j1rc_192.pkl"}
ds = PD.BuilderData("val_clean", ks=None, train=False)
rows = [int(r) for r in ds.rows]
BINS = [("<25 m", 0, 25), ("25-40 m", 25, 40), (">40 m", 40, 1e9)]
GT = {}


def gt_points(k):
    if k not in GT:
        t = PD.gt_tensors(ds.gt[rows[k]])
        gp = t["g05"][::2]
        gp = gp[(np.abs(gp).max(1) < 48.0)]
        GT[k] = gp
    return GT[k]


def seg_dist(gp, a, b):
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-6)
    t = np.clip(((gp[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(gp[:, None, :] - proj, axis=-1)  # [G, E]


for name, path in RAW.items():
    if not os.path.exists(path):
        print(name, "no cache", path)
        continue
    nodes = pickle.load(open(path, "rb"))[0]
    acc = {b[0]: dict(n=0, dead=0, rescued=0, hopeless=0, cov=0, dead_cov=0, miss=0, dead_miss=0) for b in BINS}
    for k, nd in enumerate(nodes):
        gp = gt_points(k)
        if not len(gp):
            continue
        par, valid, pi, pos = nd["parent"], nd["valid"], nd["pi"], nd["pos"]
        has = (par >= 0) & (valid > 0)
        if not has.any():
            sd_geo = np.full(len(gp), 1e9)
            dg = sd_geo
        else:
            idx = np.nonzero(has)[0]
            a, b = pos[par[idx]], pos[idx]
            d = seg_dist(gp, a, b)
            sd_geo = d.min(1)
            dg = (d + 2.0 * (1.0 - np.clip(pi[idx], 0, 1))[None]).min(1)
        r = np.linalg.norm(gp, axis=1)
        for bn, lo, hi in BINS:
            m = (r >= lo) & (r < hi)
            A = acc[bn]
            A["n"] += int(m.sum())
            A["dead"] += int((m & (dg >= 3.0)).sum())
            A["rescued"] += int((m & (dg >= 3.0) & (dg < 8.0)).sum())
            A["hopeless"] += int((m & (dg >= 8.0)).sum())
            cov = m & (sd_geo < 1.0)
            A["cov"] += int(cov.sum())
            A["dead_cov"] += int((cov & (dg >= 3.0)).sum())
            A["miss"] += int((m & (sd_geo >= 1.0)).sum())
            A["dead_miss"] += int((m & (sd_geo >= 1.0) & (dg >= 3.0)).sum())
    print(f"== {name} ({len(nodes)} frames)")
    for bn, A in acc.items():
        n = max(A["n"], 1)
        print(f"  {bn:8} points {A['n']:7}  dead {100 * A['dead'] / n:5.1f} %  rescued by clamp 8 m {100 * A['rescued'] / n:5.1f} %  hopeless {100 * A['hopeless'] / n:5.1f} %"
              f"  | covered (edge < 1 m) {100 * A['cov'] / n:5.1f} %, of them dead (price) {100 * A['dead_cov'] / max(A['cov'], 1):4.1f} %  | missed {100 * A['miss'] / n:5.1f} %, of them dead {100 * A['dead_miss'] / max(A['miss'], 1):5.1f} %")

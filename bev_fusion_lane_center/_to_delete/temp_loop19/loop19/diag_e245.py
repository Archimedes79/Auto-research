"""E245 layer-1 diagnostic (CPU, cached raw nodes of VAL 873): is the hard label tolerance what rejects far hits?

    wsl bash tools/env/_py14.sh temp/loop19/diag_e245.py

Far nodes = valid edges (parent >= 0, valid > 0) with |pos| > 40 m. A node is a hit when it lies within 1.0 m of a ground-truth lane point (g05). Rejected = path
validity pi below tau (0.7). Card kill criterion: among the rejected hits, the share whose distance to the lane lies in (0.6, 1.0] m must be >= 20 %, else the
tolerance mismatch (label d < 0.8 m, chord within 0.6 m) is not what rejects hits and the soft-target loss has nothing to fix. Also prints the same split for
all nodes and for the near range, and the share of hits per distance bin (0 - 0.3, 0.3 - 0.6, 0.6 - 1.0 m) with their mean pi.
"""
import os
import pickle

import numpy as np
from scipy.spatial import cKDTree

import p16_data as PD

HOME = os.path.expanduser("~")
RAW = {"j1": f"{HOME}/exp_gcar_b123/p18/raw_val873_e123_D7_r3_j1_192.pkl", "j1eq": f"{HOME}/exp_gcar_b123/p18/raw_val873_e18_D7_j1eq_192.pkl",
       "j1rc": f"{HOME}/exp_gcar_b123/p18/raw_val873_e18_D7_j1rc_192.pkl"}
TAU = 0.7
ds = PD.BuilderData("val_clean", ks=None, train=False)
rows = [int(r) for r in ds.rows]
trees = {}
for name, path in RAW.items():
    nodes = pickle.load(open(path, "rb"))[0]
    acc = {rng: dict(hit=0, rej=0, rej_edge=0, bins=np.zeros(3), binpi=np.zeros(3)) for rng in ("near <25 m", "mid 25-40 m", "far >40 m")}
    for k, nd in enumerate(nodes):
        if k not in trees:
            t = PD.gt_tensors(ds.gt[rows[k]])
            trees[k] = cKDTree(t["g05"]) if len(t["g05"]) else None
        if trees[k] is None:
            continue
        has = (nd["parent"] >= 0) & (nd["valid"] > 0)
        pos, pi = nd["pos"][has], nd["pi"][has]
        d = trees[k].query(pos)[0]
        r = np.linalg.norm(pos, axis=1)
        for rng, lo, hi in (("near <25 m", 0, 25), ("mid 25-40 m", 25, 40), ("far >40 m", 40, 1e9)):
            m = (r >= lo) & (r < hi) & (d < 1.0)
            A = acc[rng]
            A["hit"] += int(m.sum())
            rej = m & (pi < TAU)
            A["rej"] += int(rej.sum())
            A["rej_edge"] += int((rej & (d > 0.6)).sum())
            for i, (a, b) in enumerate(((0, 0.3), (0.3, 0.6), (0.6, 1.0))):
                s = m & (d >= a) & (d < b)
                A["bins"][i] += s.sum()
                A["binpi"][i] += pi[s].sum()
    print(f"== {name}")
    for rng, A in acc.items():
        print(f"  {rng:12} hits (node within 1 m of a lane) {A['hit']:7}  rejected (pi < {TAU}) {100 * A['rej'] / max(A['hit'], 1):5.1f} %  of the rejected, distance in (0.6, 1.0] m: {100 * A['rej_edge'] / max(A['rej'], 1):5.1f} %  "
              f"| hits by distance 0-0.3 / 0.3-0.6 / 0.6-1.0 m: {A['bins'][0]:.0f} / {A['bins'][1]:.0f} / {A['bins'][2]:.0f}, mean pi {A['binpi'][0] / max(A['bins'][0], 1):.2f} / {A['binpi'][1] / max(A['bins'][1], 1):.2f} / {A['binpi'][2] / max(A['bins'][2], 1):.2f}")

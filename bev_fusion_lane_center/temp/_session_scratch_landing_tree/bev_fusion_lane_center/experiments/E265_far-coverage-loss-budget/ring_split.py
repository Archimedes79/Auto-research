"""E266 step 0 (card author's pre-measurement, CPU only): far recall inside and outside the inference seed lattice (+-45 m).
Far GT points (>= 40 m, |x|,|y| < 48, dense 0.5 m) split by max(|x|,|y|) < 45 ("inner") or >= 45 ("ring"); R@1 of the output graph of each
registry name, and for builders also of the all-valid tree (tau 0) and the admitted tree (tau of the registry), plus the seed positions.
Usage: tools/env/_py14.sh <this file> <name> [<name> ...]"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV
import graph_eval as GE

reg = EV.registry()


def chords(r, tau):
    pos = r["pos_rev"] if "pos_rev" in r else r["pos"]
    ok = (r["valid"] > 0) & (r["pi"] >= tau) & (r["parent"] >= 0)
    return dict(pts=[np.stack([pos[r["parent"][i]], pos[i]]) for i in np.nonzero(ok)[0]], edges=[])


for name in sys.argv[1:]:
    e = reg[name]
    g, gts, info = EV.graphs_of(name, e, "val", require_cache=True)
    raw = EV.builder_raw(e, "val", require_cache=True)[0] if e["kind"] == "builder" else None
    acc = {}
    seed_ring, seed_n = 0, 0
    for k, (pred, gt) in enumerate(zip(g, gts)):
        G, _, _ = GE.dense_points(gt)
        far = np.linalg.norm(G, axis=1) >= 40.0
        if not far.any():
            continue
        Gf = G[far]
        ring = np.abs(Gf).max(1) >= 45.0
        srcs = [("output", pred)]
        if raw is not None:
            srcs += [("tree tau 0", chords(raw[k], 0.0)), (f"tree tau {e['tau']}", chords(raw[k], e["tau"]))]
            lv0 = (raw[k]["level"] == 0) & (raw[k]["valid"] > 0)
            sp = raw[k]["pos"][lv0]
            seed_n += len(sp)
            seed_ring += int((np.abs(sp).max(1) >= 45.0).sum())
        for s, gg in srcs:
            P, _, _ = GE.dense_points(gg)
            d = cKDTree(P).query(Gf)[0] if len(P) else np.full(len(Gf), np.inf)
            a = acc.setdefault(s, np.zeros(4))
            a += (np.sum((d <= 1) & ~ring), np.sum(~ring), np.sum((d <= 1) & ring), np.sum(ring))
    for s, a in acc.items():
        print(f"{name:12s} {s:12s} R@1 inner (< 45 m) {a[0] / a[1]:.4f} ({int(a[1])} pts)  ring (45 - 48 m) {a[2] / a[3]:.4f} ({int(a[3])} pts)  all far {(a[0] + a[2]) / (a[1] + a[3]):.4f}")
    if seed_n:
        print(f"{name:12s} kept seeds (level 0, valid) with max(|x|,|y|) >= 45 m: {seed_ring / seed_n:.4f} of {seed_n}")

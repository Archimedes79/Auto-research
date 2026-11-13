"""scratch (continuation of tier_view.py): the headline lanes_found = 0.63 counts a GT piece as found when >= 80 % of its points lie within 1.0 m of a delivered polyline point. Which pieces are not found? Per lane-GT piece of every VAL frame
(points inside the 48 m box, every 0.5 m): coverage = share of its points within 1.0 m of the delivered graph of baseline B + the hard close-node step (E275 read-out, full dump). Tables by tier of the piece (the lowest l4 tier of its points: 1 / 2 / 3,
0 = unlabelled counted as 3), by kind (mostly inside an intersection = more than half of its points are connector points inside an intersection; else other) and by length (< 10, 10 - 30, > 30 m): pieces, found (coverage >= 0.8),
partial (0.2 - 0.8), missing (< 0.2), length-weighted found. Also the same for the VISIBLE points only (the points a camera sees or that are near and unhidden).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/piece_found.py <seed>"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

SEED = int(sys.argv[1])
SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
rows = []                                                              # tier, inter, length, coverage_all, coverage_visible (nan without visible points), n_vis
for n, (f, d) in enumerate(zip(frames, dump)):
    ref = f.ref()
    nt = ref.nt
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    graph, bk = QE.readout(QR.apply(pred, SPEC), 0.5, 0.5)
    lines = [np.asarray(l, float) for l in graph.lines]
    tree = cKDTree(np.concatenate(lines)) if lines else None
    off = nt["off"]
    t_all = np.where(ref.ptier > 0, ref.ptier, 3)
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        P = ref.pts[a:b]
        inb = (np.abs(P) < 48.0).all(1)
        if inb.sum() < 4:
            continue
        hit = (tree.query(P)[0] <= 1.0) if tree is not None else np.zeros(len(P), bool)
        vis = ref.pvis[a:b]
        mv = inb & vis
        rows.append((int(t_all[a:b].min()), float(nt["supp"][a:b][inb].mean() > 0.5), 0.5 * float(inb.sum()), float(hit[inb].mean()), float(hit[mv].mean()) if mv.sum() >= 4 else np.nan, int(mv.sum())))
    f._lane = f._ref = None
    if (n + 1) % 200 == 0:
        print(f"  frame {n + 1}/{len(frames)}", flush=True)
R = np.array(rows)
tier, inter, length, cov, covv = R[:, 0], R[:, 1] > 0.5, R[:, 2], R[:, 3], R[:, 4]
res = dict(seed=SEED, pieces=int(len(R)))
print(f"seed {SEED}: {len(R)} GT pieces; found (coverage >= 0.8) overall {100 * (cov >= 0.8).mean():.1f} %  (length-weighted {100 * (length * (cov >= 0.8)).sum() / length.sum():.1f} %)")
print(f"{'group':44s} {'pieces':>7s} {'found':>7s} {'partial':>8s} {'missing':>8s} {'found by length':>16s} {'found, visible pts':>19s}")
for name, m in [(f"tier {t}", tier == t) for t in (1, 2, 3)] + [("mostly inside an intersection", inter), ("other pieces", ~inter)] + \
               [(f"tier {t} | {k}", (tier == t) & (inter == (k == 'intersection'))) for t in (1, 2, 3) for k in ("intersection", "other")] + \
               [("length < 10 m", length < 10), ("length 10 - 30 m", (length >= 10) & (length < 30)), ("length >= 30 m", length >= 30)]:
    if m.sum() < 5:
        continue
    cv = covv[m]
    res[name] = dict(pieces=int(m.sum()), found=float((cov[m] >= 0.8).mean()), partial=float(((cov[m] >= 0.2) & (cov[m] < 0.8)).mean()), missing=float((cov[m] < 0.2).mean()),
                     found_by_length=float((length[m] * (cov[m] >= 0.8)).sum() / length[m].sum()), found_visible=float(np.nanmean(cv >= 0.8)) if np.isfinite(cv).any() else None)
    r = res[name]
    print(f"{name:44s} {r['pieces']:7d} {100 * r['found']:6.1f}% {100 * r['partial']:7.1f}% {100 * r['missing']:7.1f}% {100 * r['found_by_length']:15.1f}% {100 * r['found_visible']:18.1f}%" if r["found_visible"] is not None else f"{name:44s} {r['pieces']:7d}")
json.dump(res, open(os.path.join(ROOT, "temp", "e284", f"piece_found_s{SEED}.json"), "w"), indent=1)

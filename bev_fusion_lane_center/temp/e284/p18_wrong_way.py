"""scratch (Mirko 2026-11-10, 'kann das System die Fahrtrichtung extrahieren?'): the median heading error of the matched nodes is 3.4 degrees, but a median hides the wrong-way nodes.
CPU, VAL 873, baseline B + hard close-node step 1.5 m (E275 read-out), both seeds. A kept delivered node is matched one-to-one to a GT node within 2.0 m (greedy, nearest first, as qgraph_eval.node_counts).
err = angle between the delivered heading and the GT heading (0 .. 180). Reported: share of matched nodes with err > 30 / > 90 / > 150 degrees (> 150 = wrong way on the same line); by the l4 tier of the GT node
(1 = the lane the car really drives, 2 = neighbours in the same direction, 3 = the rest incl. opposite lanes and branches not taken, 0 = frames without l4 labels), by camera visibility, by GT node kind
(plain / stop-exit end / other event), and by the distance to the ego (< 15, 15 - 30, > 30 m). Also the share of GT nodes that no delivered node is matched to is NOT part of this (it is node recall).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p18_wrong_way.py"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402
from lane_task import measures as MS  # noqa: E402

SPEC = QR.parse("close:1.5@0.5")
FI = QT.FI
frames = QE.val_frames()
res = {}


def table(err, mask_sets):
    out = {}
    for name, m in mask_sets.items():
        e = err[m]
        out[name] = dict(n=int(m.sum()), gt30=float((e > 30).mean()) if len(e) else None, gt90=float((e > 90).mean()) if len(e) else None, gt150=float((e > 150).mean()) if len(e) else None,
                         median=float(np.median(e)) if len(e) else None, p90=float(np.quantile(e, 0.9)) if len(e) else None)
    return out


for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    E, TIER, VIS, KIND, RNG = [], [], [], [], []
    for n, (f, d) in enumerate(zip(frames, dump)):
        ref = f.ref()
        nt = ref.nt
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
        keep, L = QE.kept(pred, 0.5, 0.5)
        idx = np.flatnonzero(keep)
        P = np.asarray(pred["nodes"]["pos"], float).reshape(-1, 2)[idx]
        H = np.asarray(pred["nodes"]["dirv"], float).reshape(-1, 2)[idx]
        G, GH = nt["pos"], nt["dirv"]
        pr = MS.match_1to1(P, G, QE.NODE_TOL)
        if len(pr):
            pi, gi = pr[:, 0], pr[:, 1]
            hn = H[pi] / (np.linalg.norm(H[pi], axis=1, keepdims=True) + 1e-9)
            E.append(np.degrees(np.arccos(np.clip((hn * GH[gi]).sum(1), -1.0, 1.0))))
            TIER.append(np.asarray(nt["tier"])[gi])
            VIS.append(np.asarray(nt["vis"])[gi])
            fl = nt["flags"][gi]
            endf = fl[:, FI["stop"]] | fl[:, FI["exit"]]
            evf = fl.any(1)
            KIND.append(np.where(endf, 1, np.where(evf, 2, 0)))
            RNG.append(np.linalg.norm(G[gi], axis=1))
        f._lane = f._ref = None
        if (n + 1) % 200 == 0:
            print(f"  seed {SEED} {n + 1}/{len(frames)}", flush=True)
    err, tier, vis, kind, rng = (np.concatenate(a) for a in (E, TIER, VIS, KIND, RNG))
    sets = {"all": np.ones(len(err), bool), "tier 1 (driven lane)": tier == 1, "tier 2 (neighbours, same direction)": tier == 2, "tier 3 (rest, opposite lanes, branches)": tier == 3, "tier 0 (no l4 label)": tier == 0,
            "camera-visible": vis.astype(bool), "not camera-visible": ~vis.astype(bool), "plain nodes": kind == 0, "stop / exit ends": kind == 1, "other events": kind == 2,
            "range < 15 m": rng < 15, "range 15 - 30 m": (rng >= 15) & (rng < 30), "range >= 30 m": rng >= 30}
    res[f"seed {SEED}"] = table(err, sets)
    print(f"seed {SEED}: matched nodes {len(err)}")
    print(f"  {'':40s}{'n':>8s}{'median':>8s}{'p90':>8s}{'>30':>8s}{'>90':>8s}{'>150':>8s}")
    for k, v in res[f"seed {SEED}"].items():
        if v["n"]:
            print(f"  {k:40s}{v['n']:8d}{v['median']:8.1f}{v['p90']:8.1f}{100 * v['gt30']:7.1f}%{100 * v['gt90']:7.1f}%{100 * v['gt150']:7.1f}%")
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "p18_wrong_way.json"), "w"), indent=1)
print("done")

"""E356: the nearest-neighbour spacing of the DELIVERED nodes of B (the stored B dump `temp/e276/predfull_e273_p_s<S>.pkl`, VAL 873 frames) before and after the close-node step of the read-out at the radius 1.5 / 2.0 / 2.5 m:
nodes per frame, the share of kept nodes that have another kept node closer than 2.0 m / 2.5 m (what D104 asks: 'no nodes closer than 2 m'), the quantiles of the nearest-neighbour distance. CPU, plain numpy (no torch).
usage: python3 temp/e356/nn_spacing.py [--seeds 0 1] [--thr 0.5]  -> table on stdout, temp/e356/nn_spacing.json"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import qgraph_readout as QR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
out = {}
print("| seed | read-out | nodes per frame (mean / p95 / max) | share with a neighbour < 2.0 m | share with a neighbour < 2.5 m | NN distance 5 % / 25 % / 50 % / 75 % (m) |")
print("|---|---|---|---|---|---|")
for S in a.seeds:
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{S}.pkl"), "rb"))
    for name, R in (("e >= thr (no step)", None), ("close:1.5", 1.5), ("close:2.0", 2.0), ("close:2.5", 2.5)):
        n_nodes = n_lt20 = n_lt25 = 0
        nn_all, per_frame = [], []
        for d in dump:
            pos = np.asarray(d["pos"], float).reshape(-1, 2)
            e = np.asarray(d["e"], float).reshape(-1)
            if R is None:
                keep = e >= a.thr
            else:
                keep = np.asarray(QR.suppress_close_nodes(pos, e, R, a.thr), bool)
            p = pos[keep]
            n_nodes += len(p)
            per_frame.append(len(p))
            if len(p) >= 2:
                dm = np.linalg.norm(p[:, None] - p[None], axis=2)
                np.fill_diagonal(dm, np.inf)
                nn = dm.min(1)
                nn_all.append(nn)
                n_lt20 += int((nn < 2.0).sum())
                n_lt25 += int((nn < 2.5).sum())
        nn_all = np.concatenate(nn_all) if nn_all else np.zeros(0)
        q = np.percentile(nn_all, [5, 25, 50, 75]) if len(nn_all) else [float("nan")] * 4
        row = dict(nodes_per_frame=n_nodes / len(dump), nodes_p95=float(np.percentile(per_frame, 95)), nodes_max=int(max(per_frame)), share_lt20=n_lt20 / max(n_nodes, 1), share_lt25=n_lt25 / max(n_nodes, 1), nn_quantiles=[float(x) for x in q], frames=len(dump))
        out[f"s{S}_{name}"] = row
        print(f"| {S} | {name} | {row['nodes_per_frame']:.1f} / {row['nodes_p95']:.0f} / {row['nodes_max']} | {row['share_lt20']:.3f} | {row['share_lt25']:.3f} | " + " / ".join(f"{x:.2f}" for x in q) + " |")
json.dump(out, open(os.path.join(ROOT, "temp", "e356", "nn_spacing.json"), "w"), indent=1)

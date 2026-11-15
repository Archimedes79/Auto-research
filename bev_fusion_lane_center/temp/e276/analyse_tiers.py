"""one-off (E276 step 6): the baseline B per tier, node level, from the nodes dumped by the E275 scoring (temp/e275/pred_<run>.pkl, exact map, candidates e >= 0.2) and the lane GT l4 of the 873 VAL
frames. CPU only; needs the l4 store to be the live one (after the merge): a frame without l4 labels stops the script.

    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh analyse_tiers.py e273_p_s0 e273_p_s1 [--thr 0.5]
writes temp/e276/tiers_<run>.json and prints a table: per tier (visible GT nodes) node recall, per zone node precision, the share of the GT nodes per tier and how many are not visible.
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

import gzip  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.5
STORE = sys.argv[sys.argv.index("--store") + 1] if "--store" in sys.argv else None            # a stored l4 folder: read the graphs directly (before the merge into the live autogt)
args = [a for a in args if a != str(thr) and a != STORE]
frames = QE.val_frames()


def graph_of(f):
    if STORE:
        return pickle.load(gzip.open(os.path.join(STORE, f.token + ".pkl.gz"), "rb"))
    return f.lane()
empty = dict(src=np.zeros(0, int), dst=np.zeros(0, int), p=np.zeros(0), long=np.zeros(0, bool), a=np.zeros(0), b=np.zeros(0))
for run in args:
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e275", f"pred_{run}.pkl"), "rb"))
    assert len(dump) == 873, len(dump)
    cs, n_nodes = [], 0
    for f, d in zip(frames, dump):
        assert d["row"] == f.row
        ref = QE.LaneRef(graph_of(f))
        assert ref.labelled or not len(ref.pts), f"frame {f.token} has no l4 labels: the live lane GT is not l4"
        pred = dict(nodes=dict(e=d["e"], pos=d["pos"], dirv=d["dirv"], flags=d["flags"], section=d["section"]), links=empty)
        c = QE.node_counts(pred, ref, thr, 0.5)
        if "tn" in c:
            cs.append(dict(tn=c["tn"], zn=c["zn"], node=np.array(c["node"], float)))
        n_nodes += int(c["node"][1])
        f._lane = f._ref = None
    p = QE.pool_counts(cs)
    tn, zn, nd = np.asarray(p["tn"]), np.asarray(p["zn"]), np.asarray(p["node"])
    out = dict(run=run, thr_e=thr, frames=len(dump), delivered_nodes=n_nodes, node_P=float(nd[0] / nd[1]), node_R=float(nd[0] / nd[2]),
               gt_nodes_per_tier=[float(tn[k][3]) for k in range(3)], gt_nodes_visible_per_tier=[float(tn[k][1]) for k in range(3)],
               node_R_visible=[float(tn[k][0] / tn[k][1]) if tn[k][1] else None for k in range(3)], node_R_all=[float(tn[k][2] / tn[k][3]) if tn[k][3] else None for k in range(3)],
               node_P_by_zone=[float(zn[k][0] / zn[k][1]) if zn[k][1] else None for k in range(4)], delivered_by_zone=[float(zn[k][1]) for k in range(4)])
    json.dump(out, open(os.path.join(ROOT, "temp", "e276", f"tiers_{run}.json"), "w"), indent=1)
    gt = np.array(out["gt_nodes_per_tier"])
    print(f"\n{run}  VAL 873, e >= {thr}: node P {out['node_P']:.3f}, node R {out['node_R']:.3f} (all tiers, within 2 m, one-to-one)")
    print("| tier | GT nodes | share | visible | recall (visible) | recall (all) | zone | delivered | precision |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in range(3):
        print(f"| {k + 1} | {gt[k]:.0f} | {100 * gt[k] / gt.sum():.1f} % | {100 * out['gt_nodes_visible_per_tier'][k] / max(gt[k], 1):.0f} % | {out['node_R_visible'][k]:.3f} | {out['node_R_all'][k]:.3f} | "
              f"{k + 1} | {out['delivered_by_zone'][k + 1]:.0f} | {out['node_P_by_zone'][k + 1]:.3f} |")
    print(f"| - | | | | | | 0 (off the lanes) | {out['delivered_by_zone'][0]:.0f} | {out['node_P_by_zone'][0]:.3f} |")

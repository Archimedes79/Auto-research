"""one-off (E274): read-out sweep on the GPU, one prediction pass per run, several read-out rules scored on the 873 VAL frames:
plain node threshold, 3 x 3 cell NMS, position NMS (greedy by e, a node within R metres of a kept node is dropped). The NMS drops nodes at the read-out (a named output step), no retraining.

    wsl bash tools/env/_py14.sh sweep_gpu.py <run> [<run> ...] [--skip-plain]
writes temp/e274/sweep_<run>.json
"""
import copy
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402


def with_e(p, e_new):
    q = copy.copy(p)
    nd = dict(p["nodes"])
    nd["e"] = e_new
    q["nodes"] = nd
    return q


def cell_nms(p):
    nd = p["nodes"]
    cell, e = np.asarray(nd["cell"]), np.asarray(nd["e"], float).reshape(-1)
    g = np.zeros((50, 50))
    g[cell[:, 0], cell[:, 1]] = e
    mx = maximum_filter(g, size=3)
    keep = e >= mx[cell[:, 0], cell[:, 1]] - 1e-12
    return with_e(p, np.where(keep, e, 0.0))


def pos_nms(p, R):
    nd = p["nodes"]
    e = np.asarray(nd["e"], float).reshape(-1)
    pos = np.asarray(nd["pos"], float).reshape(-1, 2)
    keep = np.zeros(len(e), bool)
    kept = []
    for i in np.argsort(-e):
        if e[i] < 0.2:
            break
        if kept and np.min(((pos[kept] - pos[i]) ** 2).sum(1)) < R * R:
            continue
        kept.append(i)
        keep[i] = True
    return with_e(p, np.where(keep, e, 0.0))


args = [a for a in sys.argv[1:] if not a.startswith("--")]
skip_plain = "--skip-plain" in sys.argv
frames = QE.val_frames()
for run in args:
    ds, preds = QR.predict_val(run, "exact", 0, "cuda")
    print(run, "predicted", len(preds), flush=True)
    sets = {"plain": preds, "cell3": [cell_nms(p) for p in preds], "pos1.5": [pos_nms(p, 1.5) for p in preds], "pos2.5": [pos_nms(p, 2.5) for p in preds]}
    todo = [("plain", t) for t in (0.5, 0.4, 0.3)] if not skip_plain else []
    todo += [("cell3", 0.5), ("cell3", 0.4), ("pos1.5", 0.5), ("pos1.5", 0.4), ("pos1.5", 0.3), ("pos2.5", 0.5), ("pos2.5", 0.4)]
    out = {}
    for name, te in todo:
        r = QE.score_frames(sets[name], frames, thr_e=te, thr_link=0.5)
        g, n = r["graph"], r["nodes"]
        out[f"{name}_e{te}"] = dict(topo_F1=g["topo_F1"], F1_1=g["F1@1.0"], junction_F1=g["junction_F1"], lanes_found=g["lanes_found"], Q=g["Q"], dup=g["dup_frac"], vert100_dp=g["vert100_dp"],
                                    node_P=n["node_P"], node_R=n["node_R"], link_long_P=n["link_long_P"], link_long_R=n["link_long_R"])
        print(f"{run} {name} e>={te}: topo {g['topo_F1']:.3f} F1@1 {g['F1@1.0']:.3f} junction {g['junction_F1']:.3f} lanes {g['lanes_found']:.3f} Q {g['Q']:.3f} dup {g['dup_frac']:.3f} | node P {n['node_P']:.3f} R {n['node_R']:.3f}", flush=True)
        json.dump(out, open(os.path.join(ROOT, "temp", "e274", f"sweep_{run}.json"), "w"), indent=1)

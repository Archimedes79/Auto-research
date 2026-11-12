"""one-off (E274): the read-out threshold sweep with a 3 x 3 non-maximum suppression of the node existence (a node whose e is not the largest in its 3 x 3 cell neighbourhood is dropped; ties kept), CPU only

    wsl bash tools/env/_py14.sh thr_sweep_nms.py e273_p_s0
writes temp/e274/thr_sweep_nms_<run>.json
"""
import copy
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter

ROOT = os.environ["BFLC_ROOT"]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import torch  # noqa: E402

torch.set_num_threads(2)
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402

run = sys.argv[1]
frames = QE.val_frames()
ds, preds = QR.predict_val(run, "exact", 0, "cpu", workers=1)
print("predicted", len(preds), flush=True)


def nms(p):
    q = copy.copy(p)
    nd = dict(p["nodes"])
    cell, e = np.asarray(nd["cell"]), np.asarray(nd["e"], float)
    g = np.zeros((50, 50))
    g[cell[:, 0], cell[:, 1]] = e
    mx = maximum_filter(g, size=3)
    keep = e >= mx[cell[:, 0], cell[:, 1]] - 1e-12
    nd["e"] = np.where(keep, e, 0.0)
    q["nodes"] = nd
    return q


out = {}
npreds = [nms(p) for p in preds]
for name, ps in (("nms", npreds),):
    for te in (0.5, 0.4, 0.3, 0.2):
        r = QE.score_frames(ps, frames, thr_e=te, thr_link=0.5)
        g, n = r["graph"], r["nodes"]
        out[f"{name}_e{te}"] = dict(topo_F1=g["topo_F1"], F1_1=g["F1@1.0"], junction_F1=g["junction_F1"], lanes_found=g["lanes_found"], Q=g["Q"], dup=g["dup_frac"], vert100_dp=g["vert100_dp"],
                                    node_P=n["node_P"], node_R=n["node_R"], link_long_P=n["link_long_P"], link_long_R=n["link_long_R"])
        print(f"{name} e>={te}: topo {g['topo_F1']:.3f} F1@1 {g['F1@1.0']:.3f} junction {g['junction_F1']:.3f} lanes {g['lanes_found']:.3f} dup {g['dup_frac']:.3f} vert {g['vert100_dp']:.1f} | node P {n['node_P']:.3f} R {n['node_R']:.3f}", flush=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e274", f"thr_sweep_nms_{run}.json"), "w"), indent=1)

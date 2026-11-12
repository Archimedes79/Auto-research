"""one-off (E274): which true nodes have NO delivered node within 2 m (the scorer's recall), and what distinguishes them: offset of the true node from its cell centre (corner clips), e at the cell, range, intersection, event

    wsl bash tools/env/_py14.sh diag_nodes2.py e273_p_s0 [n]
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgates as QG  # noqa: E402
import qnodes as QN  # noqa: E402
import qpredict as QP  # noqa: E402

run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
model = QP.load_model(os.path.join(ROOT, "runs", run), "cuda")
cc = QN.cell_centres("cpu")[0]                                   # (2, 50, 50)
rows = []
with torch.no_grad():
    for b in QG._batches("exact", n):
        P, no = QG._node_stats(model, b, "cuda")
        e = no["e"].float().sigmoid().cpu()
        pos = no["pos"].float().cpu()                            # (B, 2, 50, 50)
        for i in range(e.shape[0]):
            ex = b["exists"][i] > 0.5
            if not ex.any():
                continue
            deliv = pos[i][:, e[i] > 0.5].T.numpy()
            gp = (cc + b["offset"][i])[:, ex].T.numpy()           # true node positions
            off = b["offset"][i][:, ex].norm(dim=0).numpy()
            d = cKDTree(deliv).query(gp)[0] if len(deliv) else np.full(len(gp), 99.0)
            ee = e[i][ex].numpy()
            rng = gp.__pow__(2).sum(1) ** 0.5
            inter = (b["road2"][i, 1] > 0.5)[ex].numpy() if b["road2"].shape[1] > 1 else np.zeros(len(gp), bool)
            evt = (b["flags"][i].max(0)[0] > 0.5)[ex].numpy()
            rows.append(np.stack([d, off, ee, rng, inter.astype(float), evt.astype(float)], 1))
A = np.concatenate(rows)
d, off, ee, rng, inter, evt = A.T
miss = d > 2.0
out = dict(run=run, true_nodes=int(len(d)), recall_2m=float(1 - miss.mean()))
for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5)):
    m = (off >= lo) & (off < hi)
    out[f"offset {lo}-{hi} m"] = dict(n=int(m.sum()), recall_2m=float(1 - miss[m].mean()), mean_e=float(ee[m].mean()), share_e_below_0_5=float((ee[m] < 0.5).mean()))
for name, m in (("inside intersection", inter > 0.5), ("outside", inter < 0.5), ("event", evt > 0.5), ("plain", evt < 0.5), ("near <20", rng < 20), ("mid 20-35", (rng >= 20) & (rng < 35)), ("far >35", rng >= 35)):
    out[name] = dict(n=int(m.sum()), recall_2m=float(1 - miss[m].mean()), mean_e=float(ee[m].mean()))
out["missed_nodes_share_by_offset"] = {f"{lo}-{hi}": float(((off >= lo) & (off < hi) & miss).sum() / max(miss.sum(), 1)) for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5))}
out["median_distance_to_nearest_delivered_of_missed"] = float(np.median(d[miss]))
out["missed_with_delivered_within_3m"] = float((d[miss] <= 3.0).mean())
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "temp", "e274", f"diag_nodes2_{run}.json"), "w"), indent=1)

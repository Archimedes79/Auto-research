"""one-off (E274): where does the E273 node head lose recall? 300 VAL frames, cell level on the 2 m grid

    wsl bash tools/env/_py14.sh diag_nodes.py e273_p_s0 [n]
Per GT node cell: the head's existence probability e; recall at several thresholds, by range band, inside / outside the (dilated) intersection mask, for event cells and plain cells; the
share of false cells that touch a GT cell (a one-cell localisation slip, harmless at the scorer's 2 m tolerance); the histogram of e at the missed cells.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgates as QG  # noqa: E402
import qnodes as QN  # noqa: E402
import qpredict as QP  # noqa: E402

run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
dev = "cuda"
model = QP.load_model(os.path.join(ROOT, "runs", run), dev)
cc = QN.cell_centres("cpu")[0]                                    # (2, 50, 50) metres
dist = cc.norm(dim=0).numpy()
THR = (0.5, 0.4, 0.3, 0.2, 0.1)
acc = {t: dict(hit=0, tot=0, fp=0, fp_adj=0, deliv=0) for t in THR}
bands = {"near (<20 m)": dist < 20, "mid (20-35 m)": (dist >= 20) & (dist < 35), "far (>35 m)": dist >= 35}
by = {k: dict(hit=0, tot=0) for k in list(bands) + ["inside intersection mask", "outside intersection mask", "event cell", "plain cell"]}
ehist = []
with torch.no_grad():
    for b in QG._batches("exact", n):
        P, no = QG._node_stats(model, b, dev)
        e = no["e"].float().sigmoid().cpu()                      # (B, 50, 50)
        ex = b["exists"].float() > 0.5
        near = F.max_pool2d(ex.float()[:, None], 3, 1, 1)[:, 0] > 0.5      # a GT cell or its 8 neighbours
        inter = (b["road2"][:, 1] > 0.5) if b["road2"].shape[1] > 1 else torch.zeros_like(ex)
        evt = (b["flags"].float().max(1)[0] > 0.5)
        for t in THR:
            d = e > t
            a = acc[t]
            a["hit"] += int((d & ex).sum()); a["tot"] += int(ex.sum()); a["deliv"] += int(d.sum())
            fp = d & ~ex
            a["fp"] += int(fp.sum()); a["fp_adj"] += int((fp & near).sum())
        d5 = e > 0.5
        for i in range(e.shape[0]):
            for k, m in bands.items():
                mm = ex[i] & torch.from_numpy(m)
                by[k]["tot"] += int(mm.sum()); by[k]["hit"] += int((mm & d5[i]).sum())
        for k, m in (("inside intersection mask", ex & inter), ("outside intersection mask", ex & ~inter), ("event cell", ex & evt), ("plain cell", ex & ~evt)):
            by[k]["tot"] += int(m.sum()); by[k]["hit"] += int((m & d5).sum())
        ehist.append(e[ex & ~d5].numpy())
eh = np.concatenate(ehist)
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
hist = np.histogram(eh, bins=bins)[0]
out = dict(run=run, frames=n, thresholds={str(t): dict(recall=a["hit"] / a["tot"], cell_precision=a["hit"] / max(a["deliv"], 1), false_cells_per_gt_cell=a["fp"] / a["tot"], share_of_false_cells_touching_a_gt_cell=a["fp_adj"] / max(a["fp"], 1))
                                           for t, a in acc.items()},
           recall_at_0_5_by={k: v["hit"] / max(v["tot"], 1) for k, v in by.items()}, count_by={k: v["tot"] for k, v in by.items()},
           missed_cells=int(len(eh)), e_hist_of_missed={f"{bins[i]}-{bins[i + 1]}": int(hist[i]) for i in range(len(hist))})
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(ROOT, "temp", "e274"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e274", f"diag_nodes_{run}.json"), "w"), indent=1)

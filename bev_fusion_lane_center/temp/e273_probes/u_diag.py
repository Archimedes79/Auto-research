"""one-off: do the long-link endpoint scores (uout / uin) rank the GT endpoints inside the top-KL? and how good is node recall / flag recall (150 VAL frames)"""
import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import numpy as np, torch
import qgates as G, qpredict as QP
run = sys.argv[1]
KL = int(sys.argv[2]) if len(sys.argv) > 2 else 64
m = QP.load_model("/mnt/c/Auto-research/bev_fusion_lane_center/runs/" + run, "cuda").eval()
hit = np.zeros(4)   # entries total, in top-KL, exits total, in top-KL
rec = np.zeros(2)
with torch.no_grad():
    for b in G._batches("exact", 150):
        out = m(b, upto="nodes", frozen=("pyr", "nodes"), gt=False)
        no = out["no"]
        for k, (tk, name) in enumerate((("uout", "uout"), ("uin", "uin"))):
            u = no[name].sigmoid().flatten(1)
            gt = b[name].to(u.device).flatten(1) > 0.5
            e = no["e"].sigmoid().flatten(1) > 0.3
            for i in range(u.shape[0]):
                score = torch.where(e[i], u[i], torch.zeros_like(u[i]))
                top = score.topk(KL)[1]
                g = gt[i].nonzero(as_tuple=True)[0]
                hit[2 * k] += len(g); hit[2 * k + 1] += int(torch.isin(g, top).sum())
        ex = b["exists"].to(no["e"].device) > 0.5
        rec[0] += int(ex.sum()); rec[1] += int((ex & (no["e"].sigmoid() > 0.5)).sum())
print(f"top-{KL} by u: GT entries found {hit[1] / hit[0]:.3f} of {int(hit[0])}; GT exits found {hit[3] / hit[2]:.3f} of {int(hit[2])}; node recall (exact cell, e>0.5) {rec[1] / rec[0]:.3f}")

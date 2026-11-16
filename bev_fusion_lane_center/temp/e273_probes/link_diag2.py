"""one-off: GEOMETRIC link recall: a GT link u -> v counts when some predicted nodes a (within 1.5 m of u) and c (within 1.5 m of v) are joined by a predicted link; plus node recall by distance (150 VAL frames)"""
import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import numpy as np, torch
import qgates as G, qpredict as QP, qnodes as QN
run = sys.argv[1]
m = QP.load_model("/mnt/c/Auto-research/bev_fusion_lane_center/runs/" + run, "cuda").eval()
cc = QN.cell_centres("cpu")[0]                                   # (2,50,50)
stat = dict(short=np.zeros(5), long=np.zeros(5))
nodes = np.zeros(4)
with torch.no_grad():
    for b in G._batches("exact", 150):
        out = m(b, upto="links", frozen=("pyr", "nodes", "links"), gt=False)
        L, no = out["L"], out["no"]
        B = L["act"].shape[0]
        gpos = (b["offset"] + cc[None])                           # (B,2,50,50) GT node positions
        for i in range(B):
            v = L["valid"][i] & (no["e"][i].flatten().sigmoid()[L["act"][i]] > 0.5)
            pos = L["pos"][i][v].cpu().numpy()
            idx_map = -np.ones(L["pos"].shape[1], int); idx_map[v.cpu().numpy()] = np.arange(int(v.sum()))
            sel = (L["pb"] == i) & (L["logit"].sigmoid() > 0.5)
            a = idx_map[L["pa"][sel].cpu().numpy()]; c = idx_map[L["pc"][sel].cpu().numpy()]
            ok = (a >= 0) & (c >= 0)
            links = set(zip(a[ok].tolist(), c[ok].tolist()))
            lk = b["lk"][i].numpy(); lg = b["lk_long"][i].numpy()
            gp = gpos[i].permute(1, 2, 0).numpy()
            ex = (b["exists"][i] > 0.5).numpy()
            gnodes = gp[ex]
            if len(pos) and len(gnodes):
                d = np.linalg.norm(gnodes[:, None] - pos[None], axis=2).min(1)
                nodes += [len(gnodes), (d < 0.5).sum(), (d < 1.5).sum(), (d < 2.5).sum()]
            for q in range(len(lk)):
                if lk[q, 0] < 0:
                    continue
                u = gp[lk[q, 0], lk[q, 1]]; w = gp[lk[q, 2], lk[q, 3]]
                na = np.where(np.linalg.norm(pos - u, axis=1) < 1.5)[0] if len(pos) else []
                nc = np.where(np.linalg.norm(pos - w, axis=1) < 1.5)[0] if len(pos) else []
                t = stat["long" if lg[q] else "short"]
                t[0] += 1; t[1] += len(na) > 0 and len(nc) > 0
                t[2] += any((x, y) in links for x in na for y in nc)
print("GT nodes %d: predicted node within 0.5 m %.3f, 1.5 m %.3f, 2.5 m %.3f" % (nodes[0], nodes[1] / nodes[0], nodes[2] / nodes[0], nodes[3] / nodes[0]))
for n, t in stat.items():
    print(f"{n:5s}: GT links {int(t[0])}; nodes near both ends {t[1] / t[0]:.3f}; geometric link recall {t[2] / t[0]:.3f}")

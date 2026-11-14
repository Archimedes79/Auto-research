"""one-off: why is the phase 4 graph poor? link recall chain: GT link -> both nodes predicted -> candidate -> probability, short vs long (150 VAL frames)"""
import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import numpy as np, torch
import qgates as G, qpredict as QP
run = sys.argv[1]
m = QP.load_model("/mnt/c/Auto-research/bev_fusion_lane_center/runs/" + run, "cuda").eval()
tot = dict(short=np.zeros(6), long=np.zeros(6))   # gt, both nodes e>0.5, in candidates, p>0.5, p>0.3, p>0.2
fp = np.zeros(3)  # predicted links p>0.5: total, in GT, to a node that is not a GT node
with torch.no_grad():
    for b in G._batches("exact", 150):
        out = m(b, upto="links", frozen=("pyr", "nodes", "links"), gt=False)
        L, no = out["L"], out["no"]
        dev = L["act"].device
        B = L["act"].shape[0]
        ex = (no["e"].sigmoid() > 0.5).flatten(1)
        cells = L["act"]
        pr = L["logit"].sigmoid()
        key = (L["pb"] * 2500 + cells[L["pb"], L["pa"]]) * 2500 + cells[L["pb"], L["pc"]]
        order = key.argsort(); ks = key[order]; ps = pr[order]
        lk = b["lk"].to(dev); okl = lk[..., 0] >= 0
        bb = torch.arange(B, device=dev)[:, None].expand_as(okl)
        fa = lk[..., 0] * 50 + lk[..., 1]; fc = lk[..., 2] * 50 + lk[..., 3]
        gk = (bb * 2500 + fa) * 2500 + fc
        lg = b["lk_long"].to(dev)
        for name, sel in (("short", okl & ~lg), ("long", okl & lg)):
            a, c, kk, bi = fa[sel], fc[sel], gk[sel], bb[sel]
            both = ex[bi, a] & ex[bi, c]
            j = torch.searchsorted(ks, kk).clamp_max(len(ks) - 1)
            inc = ks[j] == kk
            p = torch.where(inc, ps[j], torch.zeros_like(ps[j]))
            t = tot[name]
            t[0] += int(sel.sum()); t[1] += int(both.sum()); t[2] += int((both & inc).sum())
            for q, th in ((3, 0.5), (4, 0.3), (5, 0.2)):
                t[q] += int((both & (p > th)).sum())
        hit = torch.isin(key, gk)
        sel = pr > 0.5
        fp[0] += int(sel.sum()); fp[1] += int((sel & hit).sum())
for n, t in tot.items():
    print(f"{n:5s}: GT links {int(t[0])}; both nodes found {t[1] / t[0]:.3f}; also a candidate {t[2] / t[0]:.3f}; p>0.5 {t[3] / t[0]:.3f}; p>0.3 {t[4] / t[0]:.3f}; p>0.2 {t[5] / t[0]:.3f}")
print(f"predicted links p>0.5: {int(fp[0])}, in GT {fp[1] / fp[0]:.3f}")

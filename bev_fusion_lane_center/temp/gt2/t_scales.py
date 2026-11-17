import os, sys, torch, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_gt_data as LD
LD.QData.DEFAULT_JUNCTION = True
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False)
n = fin = 0; vals = []
for i, b in enumerate(dl):
    assert b["pscale"].shape[:2] == b["pairs"].shape[:2] and b["pscale"].shape[2] == 2
    m = b["pairs"][..., 0] >= 0
    ps = b["pscale"][m]
    n += int(m.sum()); fin += int(torch.isfinite(ps[:, 0]).sum()); vals.append(ps[torch.isfinite(ps[:, 0])])
    if i >= 15: break
v = torch.cat(vals)
print("pairs", n, "with fitted scales", fin, "scale a mean", float(v[:, 0].mean()), "b mean", float(v[:, 1].mean()), "share at the grid edges (0.1 / 0.85):", float(((v == 0.05) | (v == 1.25)).float().mean()))
print("OK")

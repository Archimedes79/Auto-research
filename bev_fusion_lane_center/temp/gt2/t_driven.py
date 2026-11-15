import os, sys, torch
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_gt_data as LD
import jends as JE
LD.QData.DEFAULT_JUNCTION = True
ds, dl = LD.loader("val_clean", bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False)
n_d = n_p = 0
for i, b in enumerate(dl):
    assert b["pdrv"].shape == b["pairs"].shape[:2], (b["pdrv"].shape, b["pairs"].shape)
    n_p += int((b["pairs"][..., 0] >= 0).sum()); n_d += int(b["pdrv"].sum())
    assert (b["pdrv"][b["pairs"][..., 0] < 0] == 0).all()
    if i >= 15: break
print("pairs", n_p, "driven", n_d, "share", round(n_d / max(n_p, 1), 3))
# the loss with driven weights is finite and changes the loss
B = b["x"].shape[0]
out = dict(heat=torch.randn(B, 2, 100, 100), idx=torch.zeros(B, 2, 32, dtype=torch.long), valid=torch.zeros(B, 2, 32, dtype=torch.bool), kp_pos=torch.zeros(B, 64, 2), kp_dir=torch.zeros(B, 64, 2), kp_p=torch.zeros(B, 64), logit=torch.zeros(B, 32, 32), allowed=torch.zeros(B, 32, 32, dtype=torch.bool))
l0, _ = JE.jends_loss(out, b, driven_w=0.0)
l1, _ = JE.jends_loss(out, b, driven_w=2.0)
print("loss", float(l0), float(l1))
print("OK")

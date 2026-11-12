import os
import sys
import time

import torch

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import jmaps as JM
import lane_gt_data as LD
import qgraph as QG

LD.QData.DEFAULT_JUNCTION = True
t0 = time.time()
ds, dl = LD.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="exact", pixels=True)
print("frames", len(ds), "paint missing at start", ds.paint_missing)
b = next(iter(dl))
print("batch keys junction:", {k: tuple(b[k].shape) for k in ("jz", "jheat", "jdir", "jpaint")}, "ent", tuple(b["ent"].shape), "t", round(time.time() - t0, 1))
print("zone sum", float(b["jz"][:, 0].sum()), "core sum", float(b["jz"][:, 1].sum()), "heat max", float(b["jheat"].max()), "stop px", float(b["jpaint"][:, 0].sum()), "ped px", float(b["jpaint"][:, 1].sum()))
m = JM.JModel(width=48, d=64, ctx="attn")
n, tot = m.load_base(os.path.join(R, "runs", "e273_p_s0", "model.pt"))
print("pyramid tensors loaded", n, "of", tot)
x = b["x"].float()
P, out = m(x, joint=True)
loss, info = JM.jmaps_loss(out, b)
lp, ip = QG.pixel_loss(P, b)
(loss + lp).backward()
print("loss", float(loss), info, "pixel loss", float(lp))
kp = JM.keypoints(out, thr=0.3)
print("keypoints on an untrained head", [(len(k["ent"]), len(k["ext"])) for k in kp])
print("paint missing", ds.paint_missing, "total s", round(time.time() - t0, 1))

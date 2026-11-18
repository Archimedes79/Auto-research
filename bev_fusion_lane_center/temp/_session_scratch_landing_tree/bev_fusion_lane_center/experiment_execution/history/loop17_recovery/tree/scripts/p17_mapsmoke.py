"""Loop 17: smoke test of the map-conditioned builder: items, widening (the widened network must give the same output as the champion for zero map channels), timing of one item."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_data as PD  # noqa: E402
import p17_eval as EV  # noqa: E402
import p17_mapnet as MN  # noqa: E402

base = PD.BuilderData("train", ks=None, train=True, with_prev=False)
P = dict(np.load(f"{PD.D}/poses_train.npz"))
ds = MN.MapWrap(base, "train", P, train=True, seed=1)
t0 = time.time()
n_map = 0
for i in range(40):
    o = ds[i]
    n_map += int(o["x"][42:].abs().sum() > 0)
print("item", tuple(o["x"].shape), o["x"].dtype, f"{(time.time() - t0) / 40 * 1e3:.0f} ms per item, {n_map}/40 with a map")
net, cfg = EV.load_net("D7", "e123_D7_r3_f1", ["M=288", "K0=144"])
cfg.temporal = False
net.cfg.temporal = False
x42 = PD.collate([base[0]])["x"].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    a = net(x42, gt=None, share=0.0, prior=None)["pos"].float()
MN.widen(net)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    b = net(torch.cat([x42, torch.zeros(1, 5, 200, 200, device="cuda")], 1), gt=None, share=0.0, prior=None)["pos"].float()
print("widened vs champion, max |diff| of the node positions:", float((a - b).abs().max()))

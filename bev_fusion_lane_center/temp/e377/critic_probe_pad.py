"""critic probe: padding tokens with a type, opposite-heading twins on real TRAIN frames"""
import os, sys
import numpy as np
import torch
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for s in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jtoken as JK
import lane_ptok_data as LP
ds = LP.PTokData("train", 0, "e292_a")
idx = np.random.RandomState(2).choice(len(ds), 200, replace=False)
bt = LP.collate([ds[int(i)] for i in idx])
v, yt = bt["valid"].bool(), bt["ytype"].long()
pos, dirv = bt["pos"].float(), bt["dirv"].float()
print("padding tokens with ytype != 0:", int((~v & (yt != 0)).sum()), "of", int((~v).sum()), "padding tokens; frames with K full:", int(v.all(1).sum()))
d2 = (pos[:, :, None] - pos[:, None]).square().sum(-1)
cos = (dirv[:, :, None] * dirv[:, None]).sum(-1)
asg = v & (yt != 0)
opp = v & (yt == 0) & ((d2 <= 4.0) & (cos <= -0.5) & asg[:, None, :]).any(-1)
near = JK.near_none(pos, dirv, v, yt)
print("none tokens within 2 m of an assigned token with cos <= -0.5 per frame:", float(opp.sum(1).float().mean()), "; of them not near by the label:", float((opp & ~near).sum(1).float().mean()), "; near per frame", float(near.sum(1).float().mean()))

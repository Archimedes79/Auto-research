"""scratch (E282 critic pass): size of the cached pyramid outputs per frame and the time of NodeHead fwd+bwd on the CPU (2 threads), batch 4"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD
import qgraph as QG
import qnodes as QN
ds = LD.QData("val_clean", pixels=False, map_mode="exact")
model = QG.QGraph(width=48)
sd = torch.load(os.path.join(ROOT, "runs", "e273_p_s0_p2", "model.pt"), map_location="cpu")["state"]
own = model.state_dict(); model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
model.eval()
x = torch.from_numpy(np.asarray(ds[0]["x"], np.float32))[None]
t = time.time()
with torch.no_grad():
    P = model.pyr(x)
print("pyramid forward 1 frame", round(time.time() - t, 2), "s")
need = ("feat2", "feat1", "feat05", "road2", "centre1")
tot = 0
for k in need:
    print(k, tuple(P[k].shape), P[k].dtype)
    tot += P[k].numel()
print("floats per frame", tot, "MB fp32", tot * 4 / 1e6, "MB fp16", tot * 2 / 1e6, " 709 frames fp16 GB", 709 * tot * 2 / 1e9, "fp32 GB", 709 * tot * 4 / 1e9)
head = model.nodes
head.train()
B = 4
inp = [P[k].detach().expand(B, *P[k].shape[1:]).contiguous() for k in need]
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
for i in range(4):
    t = time.time()
    no = head(*inp)
    loss = no["e"].square().mean() + no["off"].square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    print("step", i, round(time.time() - t, 2), "s")

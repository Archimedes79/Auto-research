"""scratch: the end head end to end on 2 real VAL frames with junctions (CPU or GPU): data -> frozen B stages -> end head -> loss -> backward -> counts. Not a card; a smoke of the E284 code path."""
import os
import sys

import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qendnet as QEN  # noqa: E402
import qgraph as QG  # noqa: E402

dev = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu"
ds = LD.QData("val_clean", map_mode="error", pixels=False, ends=True)
pick = [i for i in range(0, len(ds), 7) if len(ds[i]["ent"]) > 0][:3]
print("device", dev, "frames", pick, "ends", [(len(ds[i]["ent"]), len(ds[i]["ext"]), len(ds[i]["pairs"])) for i in pick])
b = LD.collate([ds[i] for i in pick])
m = QG.QGraph(ends=True)
sd = torch.load(os.path.join(ROOT, "runs", "e273_p_s0", "model.pt"), map_location="cpu")["state"]
res = m.load_state_dict(sd, strict=False)
print("missing", sorted({k.split('.')[0] for k in res.missing_keys}), "unexpected", len(res.unexpected_keys))
m = m.to(dev)
for p in m.parameters():
    p.requires_grad_(False)
ps = QG.trainable_params(m, 8)
for p in ps:
    p.requires_grad_(True)
print("trainable M", sum(p.numel() for p in ps) / 1e6)
loss, info = QG.qgraph_loss(m, b, 8)
loss.backward()
print("loss", float(loss), info)
assert all(p.grad is not None for p in ps if p.requires_grad), "dead parameter"
out = m.ends_forward(b, gt=False)
print(QEN.end_counts(out, b))
print("smoke ok")

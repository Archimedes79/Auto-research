"""CPU smoke of the new training options on REAL frames (2 batches of 2): --capacity, --e2e, --raster exact, and their sum; one optimiser step each, the loss must be finite and the gradients where the card says."""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch.set_num_threads(1)
import jends as JE  # noqa: E402
import jmaps as JM  # noqa: E402
import jends_train as JT  # noqa: E402
import lane_gt_data as LD  # noqa: E402

LD.QData.DEFAULT_JUNCTION = True
ck = torch.load(os.path.join(ROOT, "runs", "e286_j1_s0", "model.pt"), map_location="cpu")
for raster in ("legacy", "exact"):
    if raster == "exact":
        os.environ["BFLC_RASTER"] = "exact"
    ds, dl = LD.loader("train", bs=2, workers=0, shuffle=True, map_mode="train", pixels=False)
    it = iter(dl)
    batches = [next(it), next(it)]
    for capacity, e2e in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(0)
        m = JT.JEndModel(ck["args"], order=True, evidence=True, scales=False, e2e=e2e, capacity=capacity)
        m.jmodel.load_state_dict(ck["state"])
        for p in m.parameters():
            p.requires_grad_(False)
        ps = list(m.je.parameters()) + (list(m.jmodel.jm.parameters()) if e2e else [])
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.Adam(ps, lr=1e-3)
        for b in batches:
            x = b["x"].float()
            gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
            _, jout, out = m(x, gt=gt)
            loss, info = JE.jends_loss(out, b, cross_w=1.0)
            if e2e:
                lm, im = JM.jmaps_loss(jout, b)
                loss = loss + lm
            assert torch.isfinite(loss), (raster, capacity, e2e, info)
            opt.zero_grad()
            loss.backward()
            opt.step()
        gjm = sum(float(p.grad.abs().sum()) for p in m.jmodel.jm.parameters() if p.grad is not None)
        print(f"raster {raster:6s} capacity {capacity!s:5s} e2e {e2e!s:5s}: loss {float(loss):.3f} zone targets {int(b['jz'][:, 0].sum())} cells, jm grad {gjm:.2e}", {k: round(v, 3) for k, v in info.items() if k in ("bce", "deg_ce", "deg_acc", "pos_pairs")}, flush=True)
        assert (gjm > 0) == e2e
print("CPU smoke ok")

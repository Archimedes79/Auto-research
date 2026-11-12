"""scratch (E276 step 5): one forward + backward of the lane head on REAL frames of the l4 store on the CPU, for every phase, with the tier switches (arm t) and with the sparse target (arm s): the data path
(l4 labels, tier / vis / ztier maps, sparse nodes), the loss switches and the gradient, before any GPU time is spent. The trainer itself needs CUDA; this uses its pieces.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh cpu_step.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qgraph as QG  # noqa: E402

torch.manual_seed(0)
for arm, sparse in (("t", None), ("s", (0.3, 7.5))):
    LD.QData.DEFAULT_SPARSE = sparse
    QG.CFG.update(tier_w=(3.0, 1.5, 1.0), vis_w=0.0, neg_w=(2.0, 1.5, 1.0, 1.0), link_tier_w=(3.0, 1.5, 1.0))
    ds, dl = LD.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="train")
    b = next(iter(dl))
    print(f"arm {arm}: batch tier maps: nodes {int(b['exists'].sum())}, tier 1 / 2 / 3 nodes {[int((b['tier'] == k).sum()) for k in (1, 2, 3)]}, demanded {int((b['vis'] * b['exists']).sum())} of {int(b['exists'].sum())}, "
          f"zone cells {[int((b['ztier'] == k).sum()) for k in (0, 1, 2, 3)]}", flush=True)
    assert int((b["tier"] > 0).sum()) == int(b["exists"].sum()), "every node has a tier (the store is l4)"
    model = QG.QGraph()
    sd = torch.load(os.path.join(ROOT, "runs", "e273_p_s0_p2", "model.pt"), map_location="cpu")["state"]
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    for phase in (3, 4, 5, 6, 7):
        model.zero_grad()
        loss, info = QG.qgraph_loss(model, b, phase)
        loss.backward()
        gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and np.isfinite(gn) and gn > 0, (phase, float(loss), gn)
        print(f"  phase {phase}: loss {float(loss):.3f}, gradient norm {gn:.2f}", flush=True)
print("cpu_step ok")

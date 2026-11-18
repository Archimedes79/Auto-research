"""E372 S4 step-0 probe (scratch, CPU): the size of the head loss's gradient INTO the live token inputs on real VAL frames (Fable's coder: the gradient into pos through `jtok.pair_geometry` grows as 1/L for close pairs, coincident valid
tokens ~1e6 x). The champion head (a run under runs/, default e373_cb_s0) with live_inputs=True on the live tokens of B' (runs/e292_a_s<seed>); per input field: the gradient norm, the maximum, the ratio max / median over the valid tokens; for pos: the largest
|grad| tokens with the distance to their nearest valid neighbour. usage: python pos_grad.py [--seed 0] [--n 8] [--head e373_cb_s0] [--bs 4]"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/grid", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import jtoken as JK  # noqa: E402
import jtoken_train as JTR  # noqa: E402
import lane_live_tok as LL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--head", default="e373_cb_s0")
ap.add_argument("--bs", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(2)
ds = LL.LiveData("val", a.seed)
src = LL.LiveSource(f"runs/e292_a_s{a.seed}", 512, True)
hd = JTR.load_model(a.head, "cpu")[0]
hd.live_inputs = True
rows = {k: [] for k in LL.GRAPH_KEYS}
posinfo = []
gn_nodes = []
for i0 in range(0, a.n, a.bs):
    b = LL.collate([ds[i] for i in range(i0, min(i0 + a.bs, a.n))])
    out = src(b["x"], b["table"], b["token"], b["row"])
    bt = out["batch"]
    for k in LL.GRAPH_KEYS:
        bt[k].retain_grad()
    loss, info = JK.jtoken_loss(hd(bt), bt)
    loss.backward()
    gn_nodes.append(float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in src.trainable()))))
    valid = bt["valid"].bool()
    for k in LL.GRAPH_KEYS:
        gm = bt[k].grad.abs().reshape(valid.shape[0], valid.shape[1], -1).amax(-1)
        rows[k].append(gm[valid].numpy())
    pos = bt["pos"].detach()
    gp = bt["pos"].grad.abs().amax(-1)
    for f in range(valid.shape[0]):
        v = valid[f].nonzero().flatten()
        p = pos[f][v]
        d = torch.cdist(p, p) + torch.eye(len(v)) * 1e9
        nn_d = d.min(1).values
        top = torch.argsort(gp[f][v], descending=True)[:3]
        posinfo.append([(float(gp[f][v][t]), float(nn_d[t])) for t in top] + [float(nn_d.min())])
    print(f"batch from frame {i0}: head loss {float(loss):.3f}, node-head grad norm from the head's loss {gn_nodes[-1]:.3f}", flush=True)
    for p_ in src.trainable():
        p_.grad = None
print(f"head {a.head}, B' seed {a.seed}, {a.n} VAL frames, batches of {a.bs}: gradient into the live token inputs (valid tokens)")
for k in LL.GRAPH_KEYS:
    r = np.concatenate(rows[k])
    print(f"  {k:6s} n {len(r)}  mean {r.mean():.2e}  median {np.median(r):.2e}  p99 {np.percentile(r, 99):.2e}  max {r.max():.2e}  max/median {r.max() / max(np.median(r), 1e-30):.1f}")
print("  pos: the 3 largest |grad| per frame with the distance to the nearest valid neighbour (m), and the frame's smallest neighbour distance:")
for f, row in enumerate(posinfo):
    print(f"   frame {f}: " + ", ".join(f"|g| {g:.2e} nn {d:.2f}" for g, d in row[:3]) + f" | min nn distance {row[3]:.3f}")

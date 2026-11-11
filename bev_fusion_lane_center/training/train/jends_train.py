"""Trainer of the junction pair head v2 (E287, D82) on the frozen junction maps head of E286; start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_py14.sh jends_train.py --name e287_p2_s0 --seed 0 --maps runs/e286_j1_s0/model.pt [--no-order] [--no-evidence] [--cross-w 1.0] [--steps 3000] [--smoke]

The maps run (`jmaps_train.py`, JModel = B's pyramid + the junction maps head) is frozen; only `jends.JEnds` trains (keypoints from the entry / exit heat, pair logits with the rank features of the order rule and the
evidence along the Hermite curve, loss = BCE + dual softmax + the soft crossing penalty). Output `runs/<name>/model.pt` (the JModel and the JEnds state + args), read by `evaluation/jends_eval.py`.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jends as JE  # noqa: E402
import jmaps as JM  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import trainer as T  # noqa: E402


class JEndModel(nn.Module):
    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False):
        super().__init__()
        self.jmodel = JM.JModel(width=48, d=maps_args.get("d", 64), ctx=maps_args.get("ctx", "dil"))
        self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get("d", 64), scales=scales)

    def forward(self, x, gt=None):
        with torch.no_grad():
            P, out = self.jmodel(x, joint=False)
        return P, out, self.je(out, P, gt=gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", default=None, help="the junction maps run (default runs/e286_j1_s<seed>/model.pt)")
    ap.add_argument("--no-order", action="store_true")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--cross-w", type=float, default=1.0)
    ap.add_argument("--scales", action="store_true", help="E303: two handle scales per pair (the delivered curve), trained against the fitted scales of the true pairs")
    ap.add_argument("--driven-w", type=float, default=0.0, help="E298: the positive pair terms of the pairs the ego drives count 1 + w times")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    T.seed_all(a.seed)
    LD.QData.DEFAULT_JUNCTION = True
    mp = a.maps or os.path.join(ROOT, "runs", f"e286_j1_s{a.seed}", "model.pt")
    ck = torch.load(mp, map_location="cpu")
    model = JEndModel(ck["args"], order=not a.no_order, evidence=not a.no_evidence, scales=a.scales)
    model.jmodel.load_state_dict(ck["state"])
    ds, dl = LD.loader("train", bs=a.bs, workers=a.workers, shuffle=True, map_mode="train", pixels=False)
    run = T.Run(a.name)
    steps = 20 if a.smoke else a.steps
    ps = list(model.je.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    run.log(f"jends order={not a.no_order} evidence={not a.no_evidence}: {steps} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M, maps {mp}, train frames {len(ds)} (missing lane GT: {ds.missing})")

    def loss_fn(m, b):
        dev = next(m.parameters()).device
        x = b["x"].to(dev, non_blocking=True).float()
        gt = {k: b[k].to(dev) for k in ("ent", "ext", "ent_m", "ext_m")}
        _, _, out = m(x, gt=gt)
        loss, info = JE.jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w)
        return loss, info

    T.fit(model, loss_fn, run, steps=steps, loader=dl, lr=a.lr, ckpt_every=500, seed=a.seed, args=vars(a), params=ps, config=dict(design="jends", order=not a.no_order, evidence=not a.no_evidence, maps=mp))
    torch.save(dict(state=model.state_dict(), args=dict(vars(a), maps_args=ck["args"])), run.path("model.pt"))
    run.mark_done()
    run.log("done", a.name)


if __name__ == "__main__":
    main()

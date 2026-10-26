"""One-off: is the heading flip in the trained network or in the decoding? For a few val and train batches of a run's
last checkpoint, under the EMA and the raw weights: the object head's rot loss term (as trained) and, at the GT centre cells
of the vehicles, the mean predicted (sin, cos) direction against the target's, and the share with cos(pred) cos(target) < 0."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")
from evaluation.evaluate_run import load_run  # noqa: E402
from losses import build_layer_targets, object_head_terms  # noqa: E402
from object_head import _gather  # noqa: E402
from train import forward_loss, make_datasets, make_loader, val_subset  # noqa: E402

run = sys.argv[1]
dev = "cuda"
cfg, model, loss, step, ckpt = load_run(run, dev)
ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
raw_state = {k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}
cfg.workers = 2
train_ds, val_ds = make_datasets(cfg)
for wname in ("EMA", "raw"):
    if wname == "raw":
        model.load_state_dict({**model.state_dict(), **raw_state})
    for split, ds in (("val", val_ds), ("train", train_ds)):
        loader = make_loader(val_subset(ds, 48), cfg, False)
        rl, flips, n, csum = [], 0, 0, np.zeros(4)
        with torch.no_grad():
            for batch in loader:
                out, res, gts = forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=False)
                rl.append(float(out.get("objhead.rot@+0.0", float("nan"))))
                p = res["out"][0.0]
                T = build_layer_targets(gts[0.0], loss.cfg, None, 0.0, dev)
                _, t = object_head_terms(p["obj"], T["boxes"], T["w_obj"], model.grid, loss.groups)
                ind, m = t["ind"].to(dev), (t["mask"] * t["is_box"]).to(dev) > 0
                pr = _gather(p["obj"]["rot"].float(), ind)[m]
                pr = pr / pr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                tg = t["rot"].to(dev)[m]
                flips += int(((pr[:, 1] * tg[:, 1]) < 0).sum())
                n += len(tg)
                csum += torch.cat([pr.mean(0), tg.mean(0)]).cpu().numpy() * len(tg)
        print(f"{wname:4s} {split:5s}: rot loss {np.nanmean(rl):.3f}, {n} boxes, cos sign flipped {flips / max(n, 1):.2f}, "
              f"mean pred (sin, cos) {csum[0] / n:+.2f} {csum[1] / n:+.2f}, mean target {csum[2] / n:+.2f} {csum[3] / n:+.2f}",
              flush=True)

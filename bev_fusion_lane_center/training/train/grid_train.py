"""Trainer of the grid head (E271): the 1 m centre-lane head + differentiable sparsification on the frozen BEV features, on the shared harness `training/trainer.py`.

    wsl bash tools/env/_py14.sh grid_train.py --name e271_a_s0 --seed 0 [--steps 4000] [--bs 8] [--lr 1e-3] [--width 48] [--budget 12.8] [--smoke]

runs/<name>/{log.txt, status.json, config.json, ckpt.pt, model.pt, .done} (the harness), then the evaluation on VAL at the vertex budget(s): runs/e271/<name>/card.json
(`evaluation/vertex_eval.py`: the read-out threshold is fitted on 300 TRAIN frames, VAL is scored once with the FINAL weights, nothing is selected on VAL).
Start it through `autoresearch/tools/run.py` (planned ledger row, snapshot, GPU slot), not by hand. Launched with `--smoke` it trains 40 steps and scores 24 frames.
"""
import os
import sys

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import data as GD  # noqa: E402
import head as GH  # noqa: E402
import losses as GL  # noqa: E402
import trainer as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budget", type=float, nargs="*", default=[12.8, 8.0, 14.3], help="vertices per 100 m of GT lane for the read-out threshold; the first one is the card's, all go to the frontier")
    ap.add_argument("--per100", type=float, default=GL.PER100, help="the budget the vertex loss asks for")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 40 if a.smoke else a.steps
    T.seed_all(a.seed)
    model = GH.GridHead(width=a.width)
    ds, dl = GD.loader("train", bs=a.bs, workers=a.workers)
    run = T.Run(a.name)
    run.log(f"grid head, params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, train frames {len(ds)}, steps {steps}, batch {a.bs}, seed {a.seed}")
    dev = "cuda"

    def loss_fn(m, b):
        out = m(b["x"].to(dev, non_blocking=True).float())
        return GL.grid_loss(out, {k: b[k].to(dev) for k in ("g05", "t05", "m05")}, per100=a.per100)

    T.fit(model, loss_fn, run, steps=steps, loader=dl, lr=a.lr, ckpt_every=400, seed=a.seed, args=vars(a), config=dict(design="grid_head", width=a.width, per100=a.per100))
    import vertex_eval as VE
    n = 24 if a.smoke else 0
    card, d = VE.eval_head(model, a.name, a.seed, budgets=tuple(a.budget), device=dev, n=n, extra=dict(steps=steps, bs=a.bs, lr=a.lr, width=a.width, per100=a.per100))
    run.log("VAL", {k: round(v, 4) for k, v in card["metrics"].items()}, "thr", card["thr"], "frontier", {b: {k: round(v, 3) for k, v in f.items()} for b, f in card["frontier"].items()})


if __name__ == "__main__":
    main()

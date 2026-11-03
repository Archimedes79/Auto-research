"""Trainer of the grid chain (E272): the whole chain as one differentiable function, fitted in two phases on the shared harness `training/trainer.py`.

    wsl bash tools/env/_py14.sh chain_train.py --name e272_j_s0 --seed 0 [--steps-a 600] [--steps-b 1500] [--bs 4] [--T 3] [--smoke]

Start: the grid head of E271 (`runs/e271_a_s<seed>/model.pt`, trained and scored), the hold and the assigner at their start values.
Phase A (blocks):  the head is frozen (no_grad), the hold and the assigner are fitted on the chain loss.                       runs/<name>_blocks/
Phase B (joint):   everything is unfrozen and fine-tuned on the same loss at a lower learning rate: the gradient of the fused-map terms and of the assignment reaches the head.   runs/<name>/
Then the scoring on VAL (`evaluation/chain_eval.py`): hold on and hold off, the trained and the start-value assigner. Start it through `autoresearch/tools/run.py`, not by hand.
`--smoke`: 20 + 20 steps, 24 VAL frames.
"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import chain as CH  # noqa: E402
import chain_data as CD  # noqa: E402
import trainer as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-a", type=int, default=600)
    ap.add_argument("--steps-b", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--head", default=None, help="the E271 head to start from (default runs/e271_a_s<seed>/model.pt)")
    ap.add_argument("--budget", type=float, nargs="*", default=[11.2, 8.0, 14.3])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sa, sb = (20, 20) if a.smoke else (a.steps_a, a.steps_b)
    T.seed_all(a.seed)
    model = CH.GridChain(width=a.width)
    hp = a.head or os.path.join(ROOT, "runs", f"e271_a_s{a.seed}", "model.pt")
    model.head.load_state_dict(torch.load(hp, map_location="cpu")["state"])
    ds, dl = CD.loader("train", T=a.T, bs=a.bs, workers=a.workers)
    dev = "cuda"
    cfg = dict(design="grid_chain", width=a.width, T=a.T, head_from=os.path.relpath(hp, ROOT))

    # phase A: the blocks (the head frozen)
    run_a = T.Run(a.name + "_blocks")
    run_a.log(f"chain, params head {sum(p.numel() for p in model.head.parameters()) / 1e6:.2f} M hold {sum(p.numel() for p in model.hold.parameters())} assign {sum(p.numel() for p in model.assigner.parameters())}, "
              f"train chains {len(ds)}, T {a.T}, batch {a.bs}, seed {a.seed}, head from {cfg['head_from']}")
    for p in model.head.parameters():
        p.requires_grad_(False)
    model.train_head = False
    blocks = [p for m in (model.hold, model.assigner) for p in m.parameters()]
    T.fit(model, lambda m, b: CH.chain_loss(m, b), run_a, steps=sa, loader=dl, lr=a.lr_a, ckpt_every=200, seed=a.seed, args=vars(a), params=blocks, config=dict(cfg, phase="blocks"))

    # phase B: joint
    run = T.Run(a.name)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train_head = True
    run.log(f"joint fine-tune from the blocks phase, lr {a.lr_b}")
    T.fit(model, lambda m, b: CH.chain_loss(m, b), run, steps=sb, loader=dl, lr=a.lr_b, ckpt_every=300, seed=a.seed, args=vars(a), config=dict(cfg, phase="joint"))

    import chain_eval as CE
    n = 24 if a.smoke else 0
    cards, d = CE.eval_chain(model, a.name, a.seed, budgets=tuple(a.budget), device=dev, n=n, extra=dict(steps_a=sa, steps_b=sb, bs=a.bs, T=a.T, lr_a=a.lr_a, lr_b=a.lr_b, width=a.width))
    for v, c in cards.items():
        run.log("VAL", v, {k: round(x, 4) for k, x in c["metrics"].items()}, "thr", round(c["thr"], 4), c.get("assign", ""))


if __name__ == "__main__":
    main()

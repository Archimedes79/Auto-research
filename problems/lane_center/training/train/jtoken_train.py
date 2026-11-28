"""Trainer of the token head on the predicted-node cache of the frozen lane head B' (stage 1); a run starts it through the research worker (`training/train/lane_train.py stage1`).

    python -P -m lane_center.training.train.jtoken_train --name e365_s1_s0 --seed 0 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 [--none-weight 0.25] [--head-seed 2] [--smoke]

The model is `net/model/jtoken.JTokenHead` (the whole trainable path: B' is cached, nothing of it runs), the loss `loss_jtoken.jtoken_loss`. Batches come from `net/prep/lane_ptok_data.loader("train", seed, bs, workers, ident=)`
(imported only when the loop starts): the ONE finished cache of the frozen head `--ident` (`e292_a` = B') and the seed under `runs/e365_cache`, or a run's own cache (`--ident runs/<run>/tokens`). `--smoke` = 20 steps on synthetic batches (`jtoken.synthetic_batch`, K 384, no cache,
no loader): the plumbing test and the memory check.
The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back guard and the `clock` line are those of `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0.
The architecture is the champion's, a constant (`lane_net.ARCH`: d 64, heads 4, layers 3, ffn 128; built by `lane_net.build_head`, `lane_net.load_head` rebuilds it).
`--none-weight W` (default 1) multiplies the loss of the sources whose GT successor set is empty (target: the none slot) and the type cross-entropy of the none class (`loss_jtoken.jtoken_loss`).
`live_inputs` (the head keeps the graph of its inputs, `jtoken.JTokenHead(live_inputs=)`) has no flag here (the cached inputs carry no graph): `lane_net.build_head` reads it from a run's args (absent = False), so a stage-2 run's
model.pt rebuilds with it; not in `arch` and not in the recipe. `--head-seed N` (default: --seed): the head's initialisation and the data order use N, `--seed` keeps choosing the B' cache (args['seed'] stays the cache
seed, so the read-outs open the same cache); in the recipe.
Options that no longer exist (`lane_net.REMOVED`: the tag that keeps the code -> option -> the value a stored run must hold): a run's args that hold that value (or nothing) load and resume as before, another value is refused with the
option's name and the tag (`check_removed`). D117: the c-rule `prior` / `teacher`, `init`, the sizes d / heads / layers / ffn, `token_drop`, `none_weight_near`. The closed experiments (2026-11-28, tag v1): `pos_loss` max
(E376), distillation (E379), input noise (E381), a cache root / K of one's own (E383), message rounds and early reads (E392), the position head and its curve loss (E395).
The key-weight switch of the head counts the head's own training forwards (a buffer inside the checkpoint), so a resume or a roll-back keeps it right. The scored model is the final one (nothing is chosen on VAL).
A folder that holds another recipe is refused; the same recipe is resumed. `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop without that check, and `ckpt_every` / `smoke_k`
are TEST-ONLY arguments (a run uses 500 / 384).
"""
import argparse
import json
import os
import time

import torch

from lane_center.net.model import jtoken as JK
from lane_center.net.model import lane_net as LN
from lane_center.net.loss import loss_jtoken as LJK
from lane_center.training import trainer as T

RECIPE = ("seed", "ident", "bs", "steps", "lr", "none_weight", "head_seed")
SMOKE_K = 384                                                               # tokens per synthetic frame of the smoke run: the memory of the real K


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B'")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--none-weight", type=float, default=LN.ARCH_DEFAULTS["none_weight"], dest="none_weight", help="W: multiplies the loss of the sources with an empty G(i) (target: none) and the type cross-entropy of the none class (1 = off)")
    ap.add_argument("--head-seed", type=int, default=None, dest="head_seed", help="the seed of the head's initialisation and the data order (default: --seed); --seed keeps choosing the B' cache")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    return ap.parse_args(argv)


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's; a run of a removed option is refused (`lane_net.check_removed`)"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        LN.check_removed(c.get("args") or {}, folder)
        old = {k: (c.get("args") or {}).get(k, LN.ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before an option has its default
        assert c.get("design") == "jtoken" and old == {k: args.get(k, LN.ARCH_DEFAULTS.get(k)) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def make_loss(run, none_weight=1.0):
    """`loss_jtoken.jtoken_loss` + what the log needs: every step the peak GPU memory; at the first call the loss of the start weights; at call 50 and every 100 calls a clock line (seconds per step since the last line)"""
    n, mark = [0], [time.time(), 0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        b = LN.to_device(b, next(m.parameters()).device)
        loss, info = LJK.jtoken_loss(m(b), b, none_weight=none_weight)
        info.update(gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
        if n[0] == 1 and run is not None:                                       # the loss of the start weights (before the first update)
            run.log(f"first loss of this process: {float(loss):.4f} (head step {m._step_py - 1} at this forward; before any update of this process)")
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, "
                    f"reserved {info['gpu_res_gb']:.2f} GB, valid tokens in the largest frame {float(b['valid'].sum(1).max()):.0f}; GT links outside the gate in this batch {info['gt_outside']:.0f} of "
                    f"{info['gt_links']:.0f}, ylong against the token chord class {info['gt_class']:.0f}; loss parts nll_s {info['nll_s']:.3f} nll_l {info['nll_l']:.3f} ce1 {info['ce1']:.3f} ce3 "
                    f"{info['ce3']:.3f} scale_l1 {info.get('scale_l1', float('nan')):.3f}")
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it); --seed chooses the B' cache (the loader's seed), the head seed (--head-seed, default --seed) the initialisation and the data order"""
    hs = a.seed if getattr(a, "head_seed", None) is None else a.head_seed
    T.seed_all(hs)
    args = dict(vars(a), steps=20 if a.smoke else a.steps)
    check_start(args, os.path.join(runs, a.name))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        loader, batch_fn = None, lambda step: JK.synthetic_batch(a.bs, smoke_k, seed=hs * 100003 + step)
    else:
        from lane_center.net.prep import lane_ptok_data as LP  # the lane chat's loader (card 2a); lazily, so that the tests and --smoke need no cache
        _, loader = LP.loader("train", a.seed, a.bs, a.workers, ident=a.ident)
        batch_fn = None
    run = T.Run(a.name, root=runs)
    T.seed_all(hs)                                                          # the head's initialisation never depends on what drew before
    model = LN.build_head(args)
    run.log(f"jtoken bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model)} parameters ({JK.count_params(model) / 1e6:.3f} M: the whole head, d {LN.ARCH['d']} layers {LN.ARCH['layers']} heads {LN.ARCH['heads']} ffn {LN.ARCH['ffn']}; B' is cached), "
            f"cache seed {a.seed}, head seed {hs}{' (--head-seed)' if hs != a.seed or getattr(a, 'head_seed', None) is not None else ''}, "
            + (f"none weight {a.none_weight}, " if a.none_weight != 1.0 else "")
            + (f"synthetic batches, K {smoke_k}" if a.smoke else f"ident {a.ident}"))
    T.fit(model, make_loss(run, a.none_weight), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05,
          ckpt_every=ckpt_every, seed=hs, args=args, device=dev, config=dict(design="jtoken"))
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()

"""Trainer of the E365 token head on the predicted-node cache of the frozen lane head B' (stage S1 of the token champion); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jtoken_train.py --name e365_s1_s0 --seed 0 --cache <cache dir of B' seed 0> --steps 12000 --bs 8 --lr 1e-3 --workers 4 [--smoke]

The model is `grid/jtoken.JTokenHead` (the whole trainable path: B' is cached, nothing of it runs), the loss `jtoken.jtoken_loss`. Batches come from `data/lane_ptok_data.loader("train", seed, bs, workers, cache=<dir>)`
(the lane chat's module, imported only when the loop starts). `--smoke` = 20 steps on synthetic batches (`jtoken.synthetic_batch`, no cache, no loader): the plumbing test and the memory check.
The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back guard and the `clock` line are those of `jtok2_train.py` / `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0.
The key-weight switch of the head counts the head's own training forwards (a buffer inside the checkpoint), so a resume or a roll-back keeps it right. The scored model is the final one (nothing is chosen on VAL).
A folder that holds another recipe is refused; the same recipe is resumed. Without a GPU only `--smoke` runs.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken as JK  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "cache", "bs", "steps", "lr")
SMOKE_K = 384                                                               # tokens per synthetic frame of the smoke run: the memory of the real K


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="the predicted-node cache of B' for this seed (not needed for --smoke)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    return ap.parse_args(argv)


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}
        assert c.get("design") == "jtoken" and old == {k: args[k] for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def make_loss(run):
    """`jtoken.jtoken_loss` + what the log needs: every step the peak GPU memory; at call 50 and every 100 calls a clock line (seconds per step since the last line)"""
    n, mark = [0], [time.time(), 0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        b = to_device(b, next(m.parameters()).device)
        loss, info = JK.jtoken_loss(m(b), b)
        info.update(gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, "
                    f"reserved {info['gpu_res_gb']:.2f} GB, valid tokens in the largest frame {float(b['valid'].sum(1).max()):.0f}; GT links outside the gate in this batch {info['gt_outside']:.0f} of "
                    f"{info['gt_links']:.0f}, in the gate but in the other chord class {info['gt_class']:.0f}; loss parts nll_s {info['nll_s']:.3f} nll_l {info['nll_l']:.3f} ce1 {info['ce1']:.3f} ce3 "
                    f"{info['ce3']:.3f} scale_l1 {info.get('scale_l1', float('nan')):.3f}")
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def load_model(run, dev, runs=None):
    """-> (the head of runs/<run>/model.pt in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, "model.pt"), map_location="cpu", weights_only=False)
    m = JK.JTokenHead()
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def main(argv=None, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    assert a.smoke or a.cache, "--cache <the predicted-node cache of B' for this seed> is needed (only --smoke runs without)"
    T.seed_all(a.seed)
    args = dict(vars(a), steps=20 if a.smoke else a.steps)
    check_start(args, os.path.join(runs, a.name))
    if a.smoke:
        loader, batch_fn = None, lambda step: JK.synthetic_batch(a.bs, smoke_k, seed=a.seed * 100003 + step)
    else:
        import lane_ptok_data as LP  # the lane chat's loader (card 2a); lazily, so that the tests and --smoke need no cache
        loader, batch_fn = LP.loader("train", a.seed, a.bs, a.workers, cache=a.cache), None
    run = T.Run(a.name, root=runs)
    model = JK.JTokenHead()
    run.log(f"jtoken bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model) / 1e6:.3f} M (the whole head; B' is cached), " + ("synthetic batches, K " + str(smoke_k) if a.smoke else f"cache {a.cache}"))
    T.fit(model, make_loss(run), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05, ckpt_every=ckpt_every, seed=a.seed, args=args,
          device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jtoken"))
    run.log(f"done {a.name}")


if __name__ == "__main__":
    main()

"""Trainer of the token pair head on the ground-truth nodes (E357, stage T0 of E355, D105); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_py14.sh jtok_train.py --name e357_p_s0 --seed 0 --lattice 2.5 --steps 3000 --bs 8 --lr 1e-3 --workers 4 [--smoke]

The pyramid of the baseline `runs/e273_p_s<seed>/model.pt` is frozen and runs without gradient; only the head `grid/jtok.JTok` trains (tokens = the GT nodes of the lattice, `data/lane_tok_data.py`; loss `jtok.tok_loss`).
The node tables come from the cache (`lane_tok_data.py build-cache --split train`): a run with an incomplete cache is refused (`--smoke`, 20 steps, computes what it misses).
Output `runs/<name>/` as `trainer.fit` writes it (log.txt / train.log, log.jsonl, status.json, config.json, ckpt.pt every 500 steps with resume, model.pt = the state + args (seed, lattice, d, base), .done). The scored model is the
final one (nothing is chosen on VAL). Every logged step carries the peak GPU memory of the process so far, allocated and reserved (`gpu_gb`, `gpu_res_gb`), and the nodes of the largest frame of the batch (`nodes`); at step 50 and
every 100 steps a `clock` line with the seconds per step of the last steps (the first 10 steps of the process are warm-up), the two memory peaks and the largest frame over all steps so far: the launch gate of the card reads it.
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
import jtok as JT  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "lattice", "bs", "steps", "lr", "d")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lattice", type=float, default=2.5, help="metres between the nodes of the GT node table")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="20 steps")
    return ap.parse_args(argv)


def check_start(args, base, folder):
    """before anything is written: the baseline of the seed exists, and a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    assert os.path.exists(base), f"no baseline checkpoint {base} for seed {args['seed']}"
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}
        assert c.get("design") == "jtok" and old == {k: args[k] for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def make_loss(run):
    """`jtok.tok_loss` + what the log needs: every step the peak GPU memory and the node count of the largest frame; at call 50 and every 100 calls a clock line (seconds per step since the last line)"""
    n, mark, top = [0], [time.time(), 0], [0.0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        loss, info = JT.tok_loss(JT.forward_batch(m, b), b)
        info.update(nodes=float(b["valid"].sum(1).max()), gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))        # the two peaks are the process's: over ALL steps so far
        top[0] = max(top[0], info["nodes"])
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, reserved "
                    f"{info['gpu_res_gb']:.2f} GB, largest frame {top[0]:.0f} nodes (memory grows with the square of it: a later batch can exceed this peak); GT links outside the gate in this batch {info['gt_outside']:.0f} of {info['gt_links']:.0f}")
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def main(argv=None, runs=T.RUNS, cache=TD.CACHE):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    T.seed_all(a.seed)
    base = os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")
    args = dict(vars(a), steps=20 if a.smoke else a.steps, d=JT.D, base=base)
    check_start(args, base, os.path.join(runs, a.name))
    ds, dl = TD.loader("train", a.lattice, bs=a.bs, workers=a.workers, cache=cache)
    have = ds.cached()
    assert a.smoke or have == len(ds), f"the cache holds {have} of {len(ds)} TRAIN tables: build it first (lane_tok_data.py build-cache --split train --lattice {a.lattice:g})"
    run = T.Run(a.name, root=runs)
    model = JT.JTokModel(d=JT.D, base=base)
    ps = list(model.tok.parameters())
    run.log(f"jtok lattice {a.lattice:g} m, bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M (the head; the pyramid of {base} is frozen), "
            f"train frames {len(ds)} (missing lane GT: {ds.qd.missing}), tables in the cache {have}")
    T.fit(model, make_loss(run), run, steps=args["steps"], loader=dl, lr=a.lr, ckpt_every=500, seed=a.seed, args=args, params=ps, device="cuda" if torch.cuda.is_available() else "cpu",
          config=dict(design="jtok", lattice=a.lattice, base=base))
    run.log(f"done {a.name}")


if __name__ == "__main__":
    main()

"""Trainer of the token head with the node-type head and the per-source softmax on the ground-truth nodes (E359, the question left open by E357); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jtok2_train.py --name e359_p2_s0 --seed 0 --lattice 2.5 --steps 3000 --bs 8 --lr 1e-3 --workers 4 [--smoke]

The loop, the flags, the start check, the checkpoints (every 500 steps, resume), the run-directory contract and the `clock` line are those of `jtok_train.py` (imported: flags, recipe); what differs is the model
`grid/jtok2.JTok2Model` (the E357 head + the three node-type logits) and the loss `jtok2.tok2_loss` (the exact likelihood of its docstring). Every logged step carries the four loss parts (`nll_s`, `nll_l`, `bce_in`,
`scale_l1`), the peak GPU memory and the nodes of the largest frame. The scored model is the final one (nothing is chosen on VAL). A folder that holds another recipe is refused; the same recipe is resumed.
Without a GPU only `--smoke` runs.
"""
import json
import os
import sys
import time

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok2 as J2  # noqa: E402
import jtok_train as JTT  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import trainer as T  # noqa: E402

parse_args, RECIPE = JTT.parse_args, JTT.RECIPE


def check_start(args, base, folder):
    """before anything is written: the baseline of the seed exists, and a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    assert os.path.exists(base), f"no baseline checkpoint {base} for seed {args['seed']}"
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}
        assert c.get("design") == "jtok2" and old == {k: args[k] for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def make_loss(run):
    """`jtok2.tok2_loss` + what the log needs: every step the peak GPU memory and the node count of the largest frame; at call 50 and every 100 calls a clock line (seconds per step since the last line)"""
    n, mark, top = [0], [time.time(), 0], [0.0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        loss, info = J2.tok2_loss(J2.forward_batch(m, b), b)
        info.update(nodes=float(b["valid"].sum(1).max()), gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
        top[0] = max(top[0], info["nodes"])
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, reserved "
                    f"{info['gpu_res_gb']:.2f} GB, largest frame {top[0]:.0f} nodes (memory grows with the square of it: a later batch can exceed this peak); GT links outside the gate in this batch {info['gt_outside']:.0f} of "
                    f"{info['gt_links']:.0f}; loss parts nll_s {info['nll_s']:.3f} nll_l {info['nll_l']:.3f} bce_in {info['bce_in']:.3f} scale_l1 {info.get('scale_l1', float('nan')):.3f}")
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def main(argv=None, runs=T.RUNS, cache=TD.CACHE):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    T.seed_all(a.seed)
    base = os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")
    args = dict(vars(a), steps=20 if a.smoke else a.steps, d=J2.JT.D, base=base)
    check_start(args, base, os.path.join(runs, a.name))
    ds, dl = TD.loader("train", a.lattice, bs=a.bs, workers=a.workers, cache=cache)
    have = ds.cached()
    assert a.smoke or have == len(ds), f"the cache holds {have} of {len(ds)} TRAIN tables: build it first (lane_tok_data.py build-cache --split train --lattice {a.lattice:g})"
    run = T.Run(a.name, root=runs)
    model = J2.JTok2Model(d=J2.JT.D, base=base)
    ps = list(model.tok.parameters())
    run.log(f"jtok2 lattice {a.lattice:g} m, bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M (the head + type head; the pyramid of {base} is frozen), "
            f"train frames {len(ds)} (missing lane GT: {ds.qd.missing}), tables in the cache {have}")
    T.fit(model, make_loss(run), run, steps=args["steps"], loader=dl, lr=a.lr, ckpt_every=500, seed=a.seed, args=args, params=ps, device="cuda" if torch.cuda.is_available() else "cpu",
          config=dict(design="jtok2", lattice=a.lattice, base=base))
    run.log(f"done {a.name}")


if __name__ == "__main__":
    main()

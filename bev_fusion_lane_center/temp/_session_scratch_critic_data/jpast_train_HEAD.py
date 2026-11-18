"""Trainer of E368 (stage S2 of the token champion, D113): the E365 token head, trained on SEQUENCES of consecutive keyframes of the predicted-node cache of the frozen B', with (arm r) or without (arm z, `--no-past`) the cross-attention block
to the previous keyframe's token states (`grid/jpast.py`). Start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jpast_train.py --name e368_r_s0 --seed 0 --ident e292_a --steps 8000 --bs 4 --seq 3 --lr 1e-3 --workers 4 [--cache <cache root>] [--K 384] [--no-past] [--init-seed I] [--smoke]

A step = `--bs` sequences of `--seq` consecutive keyframes of one scene (`data/lane_past_data.loader("train")`, imported only when the loop starts; the cache must have its extension: `lane_past_data.py build`), the frames of one time step as one batch.
Step 1 of a sequence is cold; step k >= 2 (arm r) reads the state of step k - 1 = the head's own final token states x and key weights w, moved into the lane frame of step k by `jpast.warp_tokens` with the stored pose (`jpast.make_past`); the gradient
runs through the steps (no detach). The loss is the E365 loss (`jtoken.jtoken_loss`) on every frame, the mean over the `--seq` frames. Arm z runs the SAME loop and loader without the block: every frame cold. The head is built first, the block (arm r) from the
RNG state after it, so that z and r start from the same head weights; `--seed` = the cache seed (B' seed 0 / 1: the cache, the loader), `--init-seed` (default = `--seed`) = the head initialisation and the sequence order (`--init-seed 1` with `--seed 0`: the seed twin z2).
The key-weight switch: the head counts its training forwards (`jtoken.SWITCH` = 1000), here the counter is held at the step's value for the `--seq` forwards of a step and advanced ONCE per optimisation step, so the switch (of the head's own key weights and of the
past's, `make_past(learned=)`) comes after 1000 steps in both arms. `u` (the kill measure of the card, `jpast.UseMeter`: RMS of the block's update with the past against its update with the past of ANOTHER SAMPLE of the batch, over RMS(x); 0 when the block ignores the content of the past) and `u_none` (against the none key only: any content-free attention satisfies it, reported next to it) are logged as the mean of the last 100 steps (arm r; the arm needs `--bs` >= 2 for the comparison). A RESUMED run re-seeds the sampler (`trainer.fit` seeds the RNG again at its start and draws a fresh permutation of the chains for the remaining steps): it is not batch-matched with its twin arm any more; `fit` logs `resumed from ckpt.pt at step N`, report such a run's resume steps.
The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back
guard and the `clock` line are those of `jtoken_train.py` / `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0. The arm-r head needs the S2 hook of `jtoken.JTokenHead` (`past_block=`, `forward(batch, past=None)`, out['x'], out['w']): `check_hook` refuses a head
without it. `--smoke` = 20 steps on synthetic sequences (K 512, no cache): the plumbing test and the memory check. A folder that holds another recipe is refused; the same recipe is resumed. `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop
without that check; `ckpt_every` / `smoke_k` are TEST-ONLY arguments (a run uses 500 / 512). Tests: grid/test_jpast_train.py
"""
import argparse
import collections
import inspect
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jpast as JP  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")
SMOKE_K = 512                                                               # tokens per synthetic frame of the smoke run: the real K
U_WINDOW = 100                                                              # `u` is logged as the mean over the last 100 steps


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the cache seed (B' seed 0 / 1)")
    ap.add_argument("--init-seed", type=int, default=None, dest="init_seed", help="the head initialisation and the sequence order (default: --seed)")
    ap.add_argument("--cache", help="the cache ROOT (default runs/e365_cache; the loader finds the folder of the seed, the ident, K and the GT signature in it)")
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B', e273_p = plain B")
    ap.add_argument("--K", type=int, help="tokens per frame, only when the root holds more than one K for the seed")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=4, help="sequences per step")
    ap.add_argument("--seq", type=int, default=3, help="consecutive keyframes per sequence")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-past", action="store_true", dest="no_past", help="arm z: the same loader and loop, no block, every frame cold")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic sequences")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


def check_options(args):
    assert args["bs"] > 0 and args["steps"] > 0 and args["lr"] > 0, f"--bs {args['bs']} --steps {args['steps']} --lr {args['lr']}"
    assert args["seq"] >= 1 and (args["seq"] >= 2 or args["no_past"]), f"--seq {args['seq']}: arm r needs a sequence of at least 2 keyframes"
    assert args["no_past"] or args["bs"] >= 2, f"--bs {args['bs']}: arm r needs at least 2 sequences per step (the kill measure u compares with the past of another sample of the batch)"


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}
        assert c.get("design") == "jpast" and old == {k: args.get(k) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def check_hook(model):
    """the S2 hook of the head: `forward(batch, past=None)` and a `past_block` attribute (the owner of jtoken.py adds them); a head without it cannot carry arm r"""
    assert "past" in inspect.signature(model.forward).parameters and hasattr(model, "past_block"), \
        "the head has no S2 hook: `forward(batch, past=None)` and `past_block` (the block is called after self-attention layer 1 as past_block(x, pos, dirv, valid, past)) are missing in jtoken.JTokenHead"


def build_head(args):
    """the head of a run's recipe: the E365 head of the defaults (`jtoken_train.build_head`) built FIRST, so that its random initialisation is the same with and without the block; the block (arm r) from the RNG state after it"""
    head = JKT.build_head({})
    if not args.get("no_past"):
        check_hook(head)
        head.past_block = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])
    return head


def load_model(run, dev, runs=None, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, name), map_location="cpu", weights_only=False)
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def synthetic_sequences(bs, seq, K, seed):
    """the smoke batch: `seq` synthetic frames of B = bs sequences (`jtoken.synthetic_batch`: frame k of a sequence is a fresh random frame, the poses are small random rotations and translations) in the layout of `lane_past_data.collate`"""
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(bs, seq, generator=g) * 0.05
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    return dict(frames=[JK.synthetic_batch(bs, K, seed=seed * 1000 + k) for k in range(seq)], rel_R=R, rel_t=torch.randn(bs, seq, 2, generator=g) * 3.0, has_prev=torch.ones(bs, seq, dtype=torch.bool))


def sequence_loss(m, b, dev, none_weight=1.0, no_past=False):
    """the loss of one optimisation step on a batch of sequences (`lane_past_data.collate`): the mean over the frames of `jtoken.jtoken_loss`, step k of a sequence reading the state of step k - 1 (arm r); every forward starts from the counter of the
    step (`set_step(s0)`), so the head's own increment leaves it advanced once after the last one -> (loss, the list of the frames' info dicts)"""
    frames = [JKT.to_device(f, dev) for f in b["frames"]]
    R, t, hp = b["rel_R"].to(dev), b["rel_t"].to(dev), b["has_prev"].to(dev)
    s0 = m._step_py
    learned = s0 >= JK.SWITCH
    out_prev, losses, infos = None, [], []
    for k, fr in enumerate(frames):
        m.set_step(s0)
        if no_past:
            out = m(fr)
        else:
            past = {} if k == 0 else JP.make_past(out_prev, frames[k - 1], fr, R[:, k], t[:, k], learned, hp[:, k])
            out = m(fr, past)
        loss, info = JK.jtoken_loss(out, fr, none_weight=none_weight)
        losses.append(loss)
        infos.append(info)
        out_prev = out
    return sum(losses) / len(losses), infos


def make_loss(run, model, no_past=False, none_weight=1.0):
    """`sequence_loss` + what the log needs: the means of the frames' info, `u` and `u_none` (arm r: the means of the last `U_WINDOW` steps), every step the peak GPU memory, at call 50 and every 100 calls a clock line (seconds per step since the last line)"""
    n, mark = [0], [time.time(), 0]
    meter = None if no_past else JP.UseMeter(model.past_block)
    u_hist, un_hist = collections.deque(maxlen=U_WINDOW), collections.deque(maxlen=U_WINDOW)
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        loss, infos = sequence_loss(m, b, next(m.parameters()).device, none_weight, no_past)
        info = {k: float(np.mean([i[k] for i in infos if k in i])) for k in sorted(set().union(*infos))}
        if meter is not None:
            u = meter.take()
            if u is not None:
                assert u["u"] is not None, "no other sample in the batch: arm r needs --bs >= 2"
                u_hist.append(u["u"])
                un_hist.append(u["u_none"])
            info["u"], info["u_none"] = (float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))
        info.update(gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
        if n[0] == 10:
            mark[:] = [time.time(), 10]
        if n[0] == 50 or n[0] % 100 == 0:
            run.log(f"clock: {(time.time() - mark[0]) / (n[0] - mark[1]):.3f} s/step over the last {n[0] - mark[1]} steps of this process; over all {n[0]} steps so far: peak GPU allocated {info['gpu_gb']:.2f} GB, "
                    f"reserved {info['gpu_res_gb']:.2f} GB, valid tokens in the largest frame {max(float(f['valid'].sum(1).max()) for f in b['frames']):.0f}; loss parts nll_s {info['nll_s']:.3f} nll_l {info['nll_l']:.3f} "
                    f"ce1 {info['ce1']:.3f} ce3 {info['ce3']:.3f} scale_l1 {info.get('scale_l1', float('nan')):.3f}" + ("" if meter is None else f" u {info['u']:.4f} u_none {info['u_none']:.4f}"))
            mark[:] = [time.time(), n[0]]
        return loss, info
    return loss_fn


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it)"""
    init = a.seed if a.init_seed is None else a.init_seed
    T.seed_all(init)
    args = dict(vars(a), steps=20 if a.smoke else a.steps, init_seed=init)
    check_start(args, os.path.join(runs, a.name))
    if a.smoke:
        loader, batch_fn = None, lambda step: synthetic_sequences(a.bs, a.seq, smoke_k, a.seed * 100003 + step)
    else:
        import lane_past_data as LPD  # the extension of the cache and the sequence loader; lazily, so that the tests and --smoke need no cache
        _, loader = LPD.loader("train", a.seed, a.bs, a.workers, K_seq=a.seq, ident=a.ident, K=a.K, **({"cache": a.cache} if a.cache else {}))
        batch_fn = None
    run = T.Run(a.name, root=runs)
    model = build_head(args)
    run.log(f"jpast bs {a.bs} x seq {a.seq}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model) / 1e6:.3f} M ({'arm z: no block, every frame cold' if a.no_past else 'arm r: the head + PastBlock, ' + str(JK.count_params(model.past_block)) + ' parameters'}), "
            f"init seed {init}, " + (f"synthetic sequences, K {smoke_k}" if a.smoke else f"cache {a.cache or 'default root'} (seed {a.seed}), ident {a.ident}, K {a.K or 'auto'}"))
    T.fit(model, make_loss(run, model, a.no_past), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05, ckpt_every=ckpt_every, seed=init, args=args,
          device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jpast"))
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()

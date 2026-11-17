"""Trainer of the E365 token head on the predicted-node cache of the frozen lane head B' (stage S1 of the token champion); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jtoken_train.py --name e365_s1_s0 --seed 0 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 [--cache <cache root>] [--K 384] [--prior rule] [--teacher rule] [--init runs/R/model.pt] [--snapshots 0,500] [--smoke]

The model is `grid/jtoken.JTokenHead` (the whole trainable path: B' is cached, nothing of it runs), the loss `jtoken.jtoken_loss`. Batches come from `data/lane_ptok_data.loader("train", seed, bs, workers, ident=, K=,
cache=)` (imported only when the loop starts): `--cache` = the cache ROOT (default `runs/e365_cache`), `--ident` = the frozen head whose tokens are cached (`e292_a` = B', `e273_p` = plain B), `--K` only when the root
holds more than one K for the seed. `--smoke` = 20 steps on synthetic batches (`jtoken.synthetic_batch`, K 384, no cache, no loader): the plumbing test and the memory check.
The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back guard and the `clock` line are those of `jtok2_train.py` / `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0.
Architecture options (all in the recipe and in `model.pt`, `load_model` rebuilds the same head through `build_head`; the defaults are the head of the first build, bit for bit): `--d` 64, `--layers` 3, `--heads` 4, `--ffn` (default 2 x d).
`--token-drop P` removes each valid token with probability P in every training step (a new draw per step, seeded by (seed, the head's step counter, the frame token)): the removed token is not valid, has type none and loses every target pair (`drop_tokens`,
the same effect on the batch as the loader's own `token_drop`, which draws once per frame and cache seed, the same removal every epoch); the targets are NOT re-contracted over the removed tokens (GT links through a removed token are lost, not bridged).
`--none-weight W` (default 1 = the loss of before) multiplies the loss of the sources whose GT successor set is empty (target: the none slot) and the type cross-entropy of the none class (`jtoken.jtoken_loss`).
`--prior rule` (E369) builds the head with c-rule as a fixed prior (`jtoken.JTokenHead(prior=)`; in `build_head`, so `load_model` rebuilds it). `--teacher rule` (E370 stage A, a training option only, not in `build_head`):
the targets are `jtoken.rule_teacher`'s (c-rule's links inside the gate, B''s own types; no GT key is read), no scale term. `--init <model.pt>` (E370 stage B): the weights of another run of the same architecture
(refused otherwise), the head's step buffer CARRIED (the key weights stay the head's own 1 - p_none: the step-0 graph is the init's), a fresh optimiser and schedule; only at a fresh start (`ckpt.pt` takes precedence on a
resume); the recipe records the path and its sha256. `--snapshots 0,500,...`: model_step<N>.pt after N updates (the format of model.pt: `load_model(..., name=)`), never overwritten.
The key-weight switch of the head counts the head's own training forwards (a buffer inside the checkpoint), so a resume or a roll-back keeps it right. The scored model is the final one (nothing is chosen on VAL).
A folder that holds another recipe is refused; the same recipe is resumed. `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop without that check, and `ckpt_every` / `smoke_k`
are TEST-ONLY arguments (a run uses 500 / 384).
"""
import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken as JK  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "cache", "ident", "K", "bs", "steps", "lr", "d", "layers", "heads", "ffn", "token_drop", "none_weight", "prior", "teacher", "init", "init_sha256")
ARCH_DEFAULTS = dict(d=64, layers=3, heads=4, ffn=None, token_drop=0.0, none_weight=1.0, prior="none", teacher="none", init=None, init_sha256=None)     # what a run of before these options means (the head of the first build); ffn None = 2 x d
SMOKE_K = 384                                                               # tokens per synthetic frame of the smoke run: the memory of the real K


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="the cache ROOT (default runs/e365_cache; the loader finds the folder of the seed, the ident, K and the GT signature in it)")
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B', e273_p = plain B")
    ap.add_argument("--K", type=int, help="tokens per frame, only when the root holds more than one K for the seed")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--d", type=int, default=ARCH_DEFAULTS["d"], help="width of the head")
    ap.add_argument("--layers", type=int, default=ARCH_DEFAULTS["layers"], help="self-attention layers")
    ap.add_argument("--heads", type=int, default=ARCH_DEFAULTS["heads"], help="attention heads (d must be a multiple)")
    ap.add_argument("--ffn", type=int, default=ARCH_DEFAULTS["ffn"], help="feed-forward width (default 2 x d)")
    ap.add_argument("--token-drop", type=float, default=ARCH_DEFAULTS["token_drop"], dest="token_drop", help="train-time removal of each valid token with this probability, a new draw every step (seeded by seed, step, frame token)")
    ap.add_argument("--none-weight", type=float, default=ARCH_DEFAULTS["none_weight"], dest="none_weight", help="W: multiplies the loss of the sources with an empty G(i) (target: none) and the type cross-entropy of the none class (1 = off)")
    ap.add_argument("--prior", choices=("none", "rule"), default=ARCH_DEFAULTS["prior"], help="E369: c-rule as a fixed prior on the logits (part of the head)")
    ap.add_argument("--teacher", choices=("none", "rule"), default=ARCH_DEFAULTS["teacher"], help="E370 stage A: c-rule's links and B''s types as the targets (no GT read, no scale term)")
    ap.add_argument("--init", default=None, help="E370 stage B: the model.pt whose weights (and step buffer) start this run; relative to the problem folder")
    ap.add_argument("--snapshots", default="", help="comma list of update counts N: model_step<N>.pt (e.g. 0,500,1000,3000,6000)")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


def check_options(args):
    """the options of the head and the loss, before anything is built or loaded: d a multiple of the heads, a positive ffn (0 is not 'the default': leave --ffn out), layers >= 1, a token drop in [0, 1)"""
    assert args["heads"] > 0 and args["d"] > 0 and args["d"] % args["heads"] == 0, f"--d {args['d']} is not a multiple of --heads {args['heads']}"
    assert args["layers"] >= 1, f"--layers {args['layers']}"
    assert args["ffn"] is None or args["ffn"] > 0, f"--ffn {args['ffn']}: a positive width, or leave it out for 2 x d"
    assert 0.0 <= args["token_drop"] < 1.0, f"--token-drop {args['token_drop']}: a probability in [0, 1)"
    assert args.get("init") is None or os.path.isfile(_abs(args["init"])), f"--init {args.get('init')}: no such file"
    snapshot_steps(args.get("snapshots", ""))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def snapshot_steps(spec):
    """'0,500,1000' -> {0, 500, 1000} (update counts >= 0; '' -> none)"""
    steps = {int(s) for s in str(spec or "").split(",") if s.strip()}
    assert all(s >= 0 for s in steps), f"--snapshots {spec}: update counts >= 0"
    return steps


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before the architecture options has the defaults; conservative: ffn None and the explicit 2 x d are the same head but count as another recipe
        assert c.get("design") == "jtoken" and old == {k: args.get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def to_device(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_head(args):
    """the head of a run's recipe (its args: d, layers, heads, ffn, prior, with the defaults of a run of before these options): the one place that builds it, for the trainer and `load_model` (the teacher is no part of the head)"""
    d, heads, layers, ffn, prior = arch(args)
    return JK.JTokenHead(d=d, heads=heads, layers=layers, ffn=ffn, **({"prior": prior} if prior != "none" else {}))


def arch(args):
    """(d, heads, layers, ffn, prior) of a run's args, the defaults filled in (ffn None = 2 x d)"""
    g = lambda k: args.get(k) if args.get(k) is not None else ARCH_DEFAULTS[k]  # noqa: E731
    d = g("d")
    assert d % g("heads") == 0, f"--d {d} is not a multiple of --heads {g('heads')}"
    ffn = 2 * d if args.get("ffn") is None else args["ffn"]
    assert ffn > 0, f"ffn {ffn}"
    return d, g("heads"), g("layers"), ffn, g("prior")


def load_init(model, path, args):
    """--init: the weights of `path` (a model.pt of this design) into `model`, the head's step buffer included (carried); refused when its architecture is not this run's"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert arch(ck["args"]) == arch(args), f"--init {path}: architecture (d, heads, layers, ffn, prior) {arch(ck['args'])} is not this run's {arch(args)}"
    model.load_state_dict(ck["state"])


def save_snapshot(run, model, args, n):
    """model_step<n>.pt in the format of model.pt (`load_model(..., name=)` reads it); never overwritten -> True when written"""
    p = run.path(f"model_step{n}.pt")
    if os.path.exists(p):
        return False
    torch.save(dict(state=model.state_dict(), args=args, steps=n), p + ".tmp")
    os.replace(p + ".tmp", p)
    return True


def step_keep(tokens, K, p, seed, step):
    """(B, K) bool: the tokens that stay in a step: each is removed with probability `p` by a generator seeded with (seed, step, crc32 of the frame's token): a new draw every step, the same on a resume"""
    return np.stack([np.random.default_rng([seed, step, zlib.crc32(t.encode())]).random(K) >= p for t in tokens])


def drop_tokens(b, keep):
    """the batch with the tokens outside `keep` (B, K) removed, as the loader's `token_drop` does it (`lane_ptok_data.PTokData`): not valid, type none, every target pair with a removed end gone (y, ylong, yscale); the targets are NOT re-contracted
    over the removed tokens (the GT links through a removed token are lost, not bridged)"""
    keep = keep.to(b["valid"].device)
    out = dict(b, valid=b["valid"] & keep, ytype=torch.where(keep, b["ytype"], torch.zeros_like(b["ytype"])))
    pair = keep[:, :, None] & keep[:, None, :]
    out["y"] = torch.where(pair, b["y"], torch.zeros_like(b["y"]))
    out["ylong"] = b["ylong"] & pair
    out["yscale"] = torch.where(pair[..., None], b["yscale"], torch.full_like(b["yscale"], float("nan")))
    return out


def make_loss(run, token_drop=0.0, seed=0, none_weight=1.0, teacher="none"):
    """`jtoken.jtoken_loss` + what the log needs: every step the peak GPU memory; at call 50 and every 100 calls a clock line (seconds per step since the last line); token_drop > 0: `drop_tokens` with a draw of `step_keep` per step
    (the step is the head's own training-step counter, which a resume and a roll-back keep); teacher 'rule': the targets of `jtoken.rule_teacher` (no GT key read), no scale term, the rule links outside the gate counted"""
    n, mark = [0], [time.time(), 0]
    gb = lambda f: f() / 1e9 if torch.cuda.is_available() else 0.0  # noqa: E731

    def loss_fn(m, b):
        n[0] += 1
        b = to_device(b, next(m.parameters()).device)
        if token_drop > 0:
            toks = b.get("token") or [f"frame{i}" for i in range(b["valid"].shape[0])]                    # a synthetic batch has no frame tokens
            b = drop_tokens(b, torch.from_numpy(step_keep(toks, b["valid"].shape[1], token_drop, seed, m._step_py)))
        if teacher == "rule":
            out = m(b)
            tb = JK.rule_teacher(out, b)
            loss, info = JK.jtoken_loss(out, tb, scale_w=0.0, none_weight=none_weight)
            info.update(rule_links=tb["rule_links"], rule_out_gate=tb["rule_out_gate"])
        else:
            loss, info = JK.jtoken_loss(m(b), b, none_weight=none_weight)
        info.update(gpu_gb=gb(torch.cuda.max_memory_allocated), gpu_res_gb=gb(torch.cuda.max_memory_reserved))
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


def load_model(run, dev, runs=None, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args); name = model.pt or a snapshot model_step<N>.pt"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, name), map_location="cpu", weights_only=False)
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it)"""
    T.seed_all(a.seed)
    args = dict(vars(a), steps=20 if a.smoke else a.steps, init_sha256=file_sha256(_abs(a.init)) if a.init else None)
    check_start(args, os.path.join(runs, a.name))
    if a.smoke:
        mk = JK.with_tokens_np if a.teacher != "none" else (lambda b: b)
        loader, batch_fn = None, lambda step: mk(JK.synthetic_batch(a.bs, smoke_k, seed=a.seed * 100003 + step))
    else:
        import lane_ptok_data as LP  # the lane chat's loader (card 2a); lazily, so that the tests and --smoke need no cache
        _, loader = LP.loader("train", a.seed, a.bs, a.workers, ident=a.ident, K=a.K, **({"cache": a.cache} if a.cache else {}))
        batch_fn = None
    run = T.Run(a.name, root=runs)
    model = build_head(args)
    fresh = not os.path.exists(run.path("ckpt.pt"))                         # a resume: ckpt.pt takes precedence over --init (and the step-0 snapshot exists)
    if a.init and fresh:
        load_init(model, _abs(a.init), args)
    snaps = snapshot_steps(a.snapshots)
    if 0 in snaps and fresh:
        save_snapshot(run, model, args, 0)
    later = sorted(s for s in snaps if s > 0)
    snap_fn = (lambda m, n: dict(snapshot=n) if n in snaps and save_snapshot(run, m, args, n) else None) if later else None
    run.log(f"jtoken bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model) / 1e6:.3f} M (the whole head, d {args['d']} layers {args['layers']} heads {args['heads']} ffn {args['ffn'] or 2 * args['d']}; B' is cached), "
            + (f"token drop {a.token_drop} per step, " if a.token_drop > 0 else "") + (f"none weight {a.none_weight}, " if a.none_weight != 1.0 else "")
            + (f"prior {a.prior}, " if a.prior != "none" else "") + (f"teacher {a.teacher} (no GT targets, no scale term), " if a.teacher != "none" else "")
            + (f"init {a.init} (sha256 {args['init_sha256'][:12]}, head step {model._step_py}{'' if fresh else ', not loaded: resumed from ckpt.pt'}), " if a.init else "")
            + (f"snapshots {sorted(snaps)}, " if snaps else "")
            + (f"synthetic batches, K {smoke_k}" if a.smoke else f"cache {a.cache or 'default root'}, ident {a.ident}, K {a.K or 'auto'}"))
    T.fit(model, make_loss(run, a.token_drop, a.seed, a.none_weight, a.teacher), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05, ckpt_every=ckpt_every, seed=a.seed, args=args,
          device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jtoken"), eval_fn=snap_fn, eval_every=functools.reduce(math.gcd, later) if later else 0)
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()

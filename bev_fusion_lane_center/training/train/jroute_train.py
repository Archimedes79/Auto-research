"""Trainer of E389 (A6, D127): the route-association head `grid/jroute.RouteHead` on the cached B' tokens of a frame (`data/lane_route_data.loader`, the batch of `lane_ptok_data` + the SD route tokens with the protocol's map errors + the labels
on_road / on_lane); start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jroute_train.py --name e389_m_s0 --seed 0 --arm m --steps 3000 --bs 8 --lr 1e-3 --workers 4 --ident e292_a [--inputs full|geo] [--init-seed I] [--cache <cache root>] [--K 384] [--smoke]

--arm m: the map modes of `lane_route_data.loader("train", seed, bs, workers, init_seed=, mode="train", ident=, K=, cache=)` (imported only when the loop starts); --arm z: the SAME loader, loss and data order with the route of every sample forced invalid (`strip_route`:
the no-map reference reads the none key only). --inputs full (default: tok + pos, dirv, p_exist, flags) | geo (no tok: the geometry-only control). The loss is `jroute.route_loss` with the positive weights n_neg / n_pos of the first FREQ_BATCHES
batches of the loader (`estimate_pos_weight`; stored in the args of config.json, a resume reuses them). Only the keys `jroute.needed(inputs)` of a batch reach the device and the head; the first batch is checked against the data contract (`jroute.check_batch`).
--seed = the cache seed (B' seed 0 / 1), --init-seed (default --seed) = the head initialisation, the data order and the map-error draws of the loader (`init_seed`). The loop, the run-directory contract, the checkpoint every 500 steps with resume and the roll-back guard are those of `trainer.fit`: AdamW lr 1e-3,
5 % warm-up, cosine to 0, bf16 autocast on the GPU. Logged every 100 steps (`trainer.fit`'s row): the loss, bce_road / bce_lane, acc_road / acc_lane, lab_road / lab_lane = the share of the valid tokens that carry a label, pos_road / pos_lane = the share of the
labelled tokens that are positive (the last four are the means of the last 100 steps). `--smoke` = 20 steps on synthetic batches (K 512, M 64, no cache): the plumbing test and the memory check. A folder that holds another recipe is refused; the same recipe is
resumed (a resumed run re-seeds the loader order: `trainer.fit` logs `resumed from ckpt.pt at step N`). `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop without that check (`ckpt_every` / `smoke_k` are TEST-ONLY arguments).
Tests: grid/test_jroute.py
"""
import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jroute as JR  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "arm", "inputs")
SMOKE_K, SMOKE_M = 512, 160                                                 # tokens / route keys per synthetic frame of the smoke run: the real K and M
FREQ_BATCHES = 50                                                           # the batches whose labels give the positive weights
WINDOW = 100                                                                # the logged accuracies are the means of the last 100 steps


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the cache seed (B' seed 0 / 1)")
    ap.add_argument("--init-seed", type=int, default=None, dest="init_seed", help="the head initialisation and the data order (default: --seed)")
    ap.add_argument("--arm", choices=("m", "z"), default="m", help="m = with the SD route tokens, z = the no-map reference (the route of every sample forced invalid)")
    ap.add_argument("--inputs", choices=sorted(JR.INPUT_SETS), default="full", help="full = tok + the geometry, geo = the geometry only (the control)")
    ap.add_argument("--cache", help="the cache ROOT (default: the loader's)")
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B', e273_p = plain B")
    ap.add_argument("--K", type=int, help="tokens per frame, only when the root holds more than one K for the seed")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


def check_options(args):
    assert args["bs"] > 0 and args["steps"] > 0 and args["lr"] > 0, f"--bs {args['bs']} --steps {args['steps']} --lr {args['lr']}"


def recipe_of(d):
    return {k: d.get(k) for k in RECIPE}


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's -> the args of the held run (None: a new folder)"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if not held:
        return None
    c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
    old = recipe_of(c.get("args") or {})
    assert c.get("design") == "jroute" and old == recipe_of(args), f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"
    return c.get("args") or {}


def strip_route(b):
    """arm z: the batch with the route of every sample invalid (the none key only; the content of the route tokens is never read)"""
    return dict(b, route=dict(b["route"], valid=torch.zeros_like(b["route"]["valid"])))


def move(v, dev):
    return {k: move(x, dev) for k, x in v.items()} if isinstance(v, dict) else v.to(dev, non_blocking=True)


def to_device(b, inputs, dev):
    """only the keys the head and the loss read (`jroute.needed`): the dense GT targets, the labels of the cache and the python lists of the batch stay on the host"""
    return {k: move(b[k], dev) for k in JR.needed(inputs)}


def build_head(args):
    """the head of a run's args (the one place that builds it, for the trainer and `load_model`)"""
    return JR.RouteHead(inputs=args.get("inputs") or "full")


def load_model(run, dev, runs=None, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, name), map_location="cpu", weights_only=False)
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def estimate_pos_weight(batches):
    """(road, lane) = n_neg / n_pos of the labelled tokens of the batches (host batches of the loader)"""
    return JR.pos_weights([JR.label_counts(b) for b in batches])


def make_loss(arm, inputs, pos_weight):
    """the loss of one step on a host batch: to the device, the contract check at the first batch, arm z strips the route, `jroute.route_loss` with the positive weights; the info holds the step's bce_road / bce_lane and the means of the last WINDOW steps
    of acc_road / acc_lane (steps with a labelled token), lab_road / lab_lane (the labelled share of the valid tokens) and pos_road / pos_lane (the positive share of the labelled tokens)"""
    win = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW))
    n = [0]

    def loss_fn(m, b):
        n[0] += 1
        b = to_device(b, inputs, next(m.parameters()).device)
        if n[0] == 1:
            JR.check_batch(b)
        if arm == "z":
            b = strip_route(b)
        loss, info = JR.route_loss(m(b), b, pos_weight)
        nv = max(float(b["valid"].sum()), 1.0)
        for k in ("road", "lane"):
            win[f"lab_{k}"].append(info[f"n_{k}"] / nv)
            if info[f"n_{k}"] > 0:
                win[f"acc_{k}"].append(info[f"acc_{k}"])
                win[f"pos_{k}"].append(info[f"pos_{k}"] / info[f"n_{k}"])
        return loss, dict(bce_road=info["bce_road"], bce_lane=info["bce_lane"], **{k: float(np.mean(v)) for k, v in sorted(win.items())})
    return loss_fn


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it)"""
    init = a.seed if a.init_seed is None else a.init_seed
    T.seed_all(init)
    args = dict(vars(a), steps=20 if a.smoke else a.steps, init_seed=init)
    held = check_start(args, os.path.join(runs, a.name))
    if a.smoke:
        loader, batch_fn = None, lambda step: JR.synthetic_batch(a.bs, smoke_k, SMOKE_M, a.seed * 100003 + step)
        first = [batch_fn(s) for s in range(FREQ_BATCHES)]
    else:
        import lane_route_data as LRD  # the route loader (the cached B' tokens + the route tokens + the labels); lazily, so that the tests and --smoke need no cache
        ds, loader = LRD.loader("train", a.seed, a.bs, a.workers, init_seed=init, mode="train", ident=a.ident, K=a.K, **({"cache": a.cache} if a.cache else {}))
        man = ds.rc.manifest                                                                                        # the identity of the TRAIN route cache this head trains on: a read-out refuses a head on a cache of another version / lane GT / route cap
        args["route_cache"] = dict(split="train", version=man["version"], lane_gt=man.get("lane_gt"), max_samples=man["params"]["max_samples"], tokens_sha=man["tokens_sha"])
        assert not held or held.get("route_cache") in (None, args["route_cache"]), f"{a.name}: a resume on another route cache ({held.get('route_cache')} -> {args['route_cache']}): use another --name"
        batch_fn, first = None, itertools.islice(loader, FREQ_BATCHES)
    pw = held["pos_weight"] if held and held.get("pos_weight") else estimate_pos_weight(first)
    args["pos_weight"] = [float(w) for w in pw]
    run = T.Run(a.name, root=runs)
    model = build_head(args)
    run.log(f"jroute arm {a.arm} inputs {a.inputs}: {args['steps']} steps, bs {a.bs}, lr {a.lr}, trainable {sum(p.numel() for p in model.parameters())} parameters, pos_weight road {pw[0]:.3f} lane {pw[1]:.3f}, init seed {init}, "
            + (f"synthetic batches, K {smoke_k}, M {SMOKE_M}" if a.smoke else f"cache {a.cache or 'default root'} (seed {a.seed}), ident {a.ident}, K {a.K or 'auto'}"))
    T.fit(model, make_loss(a.arm, a.inputs, tuple(pw)), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05, ckpt_every=ckpt_every, seed=init, args=args,
          device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jroute"))
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()

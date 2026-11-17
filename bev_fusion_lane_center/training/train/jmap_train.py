"""Trainer of E374 (stage S3 of the token champion, D113): the E365 token head on the predicted-node cache of the frozen B', with (arm m) or without (arm z, `--no-map`) the cross-attention block to the SD-map tokens of the frame (`grid/jmap.py`). Start it through `autoresearch/tools/run.py`.

    wsl bash tools/env/_pyr.sh jmap_train.py --name e374_m_s0 --seed 0 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 [--cache <cache root>] [--K 384] [--no-map] [--map-mode train|other] [--init-seed I] [--smoke]

A step = `--bs` frames drawn by the plain sampler of `jtoken_train.py` (`data/lane_map_data.loader("train")`: the batches of `lane_ptok_data.loader` + `batch["maps"]`, mode `train`: every read of a frame draws its map from the card's mix of exact / error-like / counts unknown / absent / gross maps, seeded by (init seed, frame token,
the draw index of the frame); a RESUMED run starts the draw counters again: not bit-reproducible, report the resume steps). The loss is the E365 loss (`jtoken.jtoken_loss`); the head counts its own training forwards (one per step: the key-weight switch comes after `jtoken.SWITCH` = 1000 steps in both arms).
`--map-mode other` is the CAPACITY CONTROL of the card (arm c): the same block and loop, but every draw of the map is made on a RANDOM frame of ANOTHER scene (a fresh random choice at every read, not a fixed frame-to-map plan, which could be memorised; `lane_map_data` mode `other`): same parameters
and compute, no information about this frame; it needs the block (not with `--no-map`). Arm z runs the SAME trainer and loader with `--no-map`: no block, no map is built, the batch is `lane_ptok_data`'s bit for bit (its loss trajectory equals `jtoken_train.py`'s on the same batches: grid/test_jmap_train.py). The head is built first, the block (arm m) from the RNG state after it, so that z and m start from the same
head weights; `--seed` = the cache seed (B' seed 0 / 1: the cache, the loader), `--init-seed` (default = `--seed`) = the head initialisation, the sampler order and the map draws (`--init-seed 1` with `--seed 0`: the seed twin z2).
`v` (the kill measure of the card, `jpast.UseMeter` on the block: RMS of the block's update with the map against its update with the map of ANOTHER SAMPLE of the batch, over RMS(x); 0 when the block ignores the content of the map) and `v_none` (against the none key only: any content-free attention satisfies it, reported next to it) are logged as the mean of
the last 100 steps (arm m; it needs `--bs` >= 2 for the comparison; `trainer.fit` logs a row every 100 steps: the card's last 500 steps are five rows). The loop, the run-directory contract, the checkpoint every 500 steps with resume, the roll-back guard and the `clock` line are those of `jtoken_train.py` / `trainer.fit`; AdamW lr 1e-3, 5 % warm-up, cosine to 0.
The arm-m head needs the S3 hook of `jtoken.JTokenHead` (`map_block=`, `forward(batch, past=None, maps=None)`): `check_hook` refuses a head without it. `--smoke` = 20 steps on synthetic batches (K 512, 704 map samples = the cap, near the lane tokens, no cache): the plumbing test and the memory check. A folder that holds another recipe
is refused; the same recipe is resumed. `main` needs a GPU unless `--smoke`; `train(args, runs, ckpt_every, smoke_k)` is the loop without that check; `ckpt_every` / `smoke_k` are TEST-ONLY arguments (a run uses 500 / 512). Tests: grid/test_jmap_train.py
"""
import argparse
import collections
import inspect
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jmap as JM  # noqa: E402
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_map_data as LMD  # noqa: E402
import trainer as T  # noqa: E402

RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_map", "map_mode")
SMOKE_K = 512                                                               # tokens per synthetic frame of the smoke run: the real K
SMOKE_M = LMD.MAX_SAMPLES                                                   # map samples per synthetic frame of the smoke run: the cap of `lane_map_data` (the worst case of the memory check)
V_WINDOW = 100                                                              # `v` is logged as the mean over the last 100 steps


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the cache seed (B' seed 0 / 1)")
    ap.add_argument("--init-seed", type=int, default=None, dest="init_seed", help="the head initialisation, the sampler order and the map draws (default: --seed)")
    ap.add_argument("--cache", help="the cache ROOT (default runs/e365_cache; the loader finds the folder of the seed, the ident, K and the GT signature in it)")
    ap.add_argument("--ident", default="e292_a", help="the frozen head of the cached tokens: e292_a = B', e273_p = plain B")
    ap.add_argument("--K", type=int, help="tokens per frame, only when the root holds more than one K for the seed")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8, help="frames per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-map", action="store_true", dest="no_map", help="arm z: the same loader and loop, no block, no map is built")
    ap.add_argument("--map-mode", default="train", choices=("train", "other"), dest="map_mode", help="train = the card's mix of maps of THIS frame; other = the capacity control c: the same mix drawn on a random frame of ANOTHER scene at every read (no information)")
    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


def check_options(args):
    assert args["bs"] > 0 and args["steps"] > 0 and args["lr"] > 0, f"--bs {args['bs']} --steps {args['steps']} --lr {args['lr']}"
    assert not args["no_map"] or args["map_mode"] == "train", "--map-mode other needs the block: arm z (--no-map) builds no map"
    assert args["no_map"] or args["bs"] >= 2, f"--bs {args['bs']}: arm m needs at least 2 frames per step (the kill measure v compares with the map of another sample of the batch)"


def check_start(args, folder):
    """before anything is written: a folder that already holds a run is only RESUMED (the same recipe), never another run's"""
    held = [f for f in ("config.json", "ckpt.pt", "model.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        c = json.load(open(os.path.join(folder, "config.json"))) if "config.json" in held else {}
        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}
        assert c.get("design") == "jmap" and old == {k: args.get(k) for k in RECIPE}, f"{folder} holds {held} of another run ({c.get('design')}, {old}): use another --name"


def check_hook(model):
    """the S3 hook of the head: `forward(batch, past=None, maps=None)` and a `map_block` attribute; a head without it cannot carry arm m"""
    assert "maps" in inspect.signature(model.forward).parameters and hasattr(model, "map_block"), \
        "the head has no S3 hook: `forward(batch, past=None, maps=None)` and `map_block` (the block is called after self-attention layer 2 as map_block(x, pos, dirv, valid, maps)) are missing in jtoken.JTokenHead"


def build_head(args):
    """the head of a run's recipe: the E365 head of the defaults (`jtoken_train.build_head`) built FIRST, so that its random initialisation is the same with and without the block; the block (arm m) from the RNG state after it"""
    head = JKT.build_head({})
    if not args.get("no_map"):
        check_hook(head)
        head.map_block = JM.MapBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])
    return head


def load_model(run, dev, runs=None, name="model.pt"):
    """-> (the head of runs/<run>/<name> in eval mode, its args)"""
    ck = torch.load(os.path.join(runs or T.RUNS, run, name), map_location="cpu", weights_only=False)
    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["args"]


def synthetic_maps(b, seed, n=SMOKE_M):
    """the smoke maps of a synthetic batch: per frame n samples near its lane tokens (the lane tokens used again and again: the token's position + 0.5 m of noise, its heading, 1 - 3 lanes), in the layout of `lane_map_data.collate_maps`"""
    rng = np.random.RandomState(seed)
    tabs = []
    for f in range(b["pos"].shape[0]):
        idx = np.flatnonzero((b["ytype"][f] > 0).numpy() & b["valid"][f].numpy())[:n]
        idx = np.resize(idx, n)
        tabs.append(LMD.check_keys(dict(pos=(b["pos"][f, idx].numpy() + rng.randn(len(idx), 2) * 0.5).astype(np.float32), dir=b["dirv"][f, idx].numpy().astype(np.float32), count=(1 + idx % 3).astype(np.int64), valid=np.ones(len(idx), bool))))
    return LMD.collate_maps(tabs)


def synthetic_batch(bs, K, seed, no_map=False, other=False):
    """the smoke batch: `jtoken.synthetic_batch` (arm z: just that, as `jtoken_train.py --smoke` makes it) + synthetic maps (arm m); other (the capacity control): every frame gets the map of ANOTHER frame of the batch (rolled by one)"""
    b = JK.synthetic_batch(bs, K, seed=seed)
    if no_map:
        return b
    maps = synthetic_maps(b, seed)
    return dict(b, maps={k: v.roll(1, 0) for k, v in maps.items()} if other else maps)


class MapsView:
    """the head as `jtoken_train.make_loss` calls it, m(b), with the maps of the batch handed to the hook: m(b) = m(b, maps=maps)"""

    def __init__(self, m, maps):
        self.m, self.maps = m, maps

    def __call__(self, b):
        return self.m(b, maps=self.maps)

    def parameters(self):
        return self.m.parameters()

    @property
    def _step_py(self):
        return self.m._step_py


def make_loss(run, model, no_map=False):
    """the E365 loss and its log (`jtoken_train.make_loss`: the peak GPU memory, the first loss, the clock line) with the maps of the batch handed to the head; arm m adds `v` and `v_none` (the means of the last `V_WINDOW` steps) to the info"""
    base = JKT.make_loss(run)
    meter = None if no_map else JM.UseMeter(model.map_block)
    v_hist, vn_hist = collections.deque(maxlen=V_WINDOW), collections.deque(maxlen=V_WINDOW)

    def loss_fn(m, b):
        if no_map:
            return base(m, b)
        dev = next(m.parameters()).device
        loss, info = base(MapsView(m, {k: v.to(dev, non_blocking=True) for k, v in b["maps"].items()}), b)
        u = meter.take()
        if u is not None:
            assert u["u"] is not None, "no other sample in the batch: arm m needs --bs >= 2"
            v_hist.append(u["u"])
            vn_hist.append(u["u_none"])
        info["v"], info["v_none"] = (float(np.mean(h)) if h else 0.0 for h in (v_hist, vn_hist))
        return loss, info
    return loss_fn


def train(a, runs=T.RUNS, ckpt_every=500, smoke_k=SMOKE_K):
    """the loop of the parsed arguments `a` (no GPU check: `main` has it)"""
    init = a.seed if a.init_seed is None else a.init_seed
    T.seed_all(init)
    args = dict(vars(a), steps=20 if a.smoke else a.steps, init_seed=init)
    check_start(args, os.path.join(runs, a.name))
    if a.smoke:
        loader, batch_fn = None, lambda step: synthetic_batch(a.bs, smoke_k, init * 100003 + step, a.no_map, a.map_mode == "other")
    else:
        _, loader = LMD.loader("train", a.seed, a.bs, a.workers, init_seed=init, maps=not a.no_map, mode=a.map_mode, ident=a.ident, K=a.K, **({"cache": a.cache} if a.cache else {}))
        batch_fn = None
    run = T.Run(a.name, root=runs)
    model = build_head(args)
    run.log(f"jmap bs {a.bs}: {args['steps']} steps, lr {a.lr}, trainable {JK.count_params(model) / 1e6:.3f} M ({'arm z: no block, no map' if a.no_map else 'arm m: the head + MapBlock, ' + str(JK.count_params(model.map_block)) + ' parameters'}), "
            f"init seed {init}, map mode {a.map_mode}, " + (f"synthetic batches, K {smoke_k}" if a.smoke else f"cache {a.cache or 'default root'} (seed {a.seed}), ident {a.ident}, K {a.K or 'auto'}"))
    T.fit(model, make_loss(run, model, a.no_map), run, steps=args["steps"], loader=loader, batch_fn=batch_fn, lr=a.lr, sched="cosine", warmup=0.05, ckpt_every=ckpt_every, seed=init, args=args,
          device="cuda" if torch.cuda.is_available() else "cpu", config=dict(design="jmap"))
    run.log(f"done {a.name}")


def main(argv=None, runs=T.RUNS):
    a = parse_args(argv)
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke (20 steps, a plumbing test) may run on the CPU"
    train(a, runs)


if __name__ == "__main__":
    main()

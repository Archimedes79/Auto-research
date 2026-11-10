"""In-process A/B timing of the speed switches of a training step (follow-up of D53 / D54; Mirko 2026-11-07: "maybe you can measure timing on less?").

A timing run through run.py builds the model and the data and saves a checkpoint for every setting (3 - 7 minutes, of which the steps are 4) and sees the drift of the
machine between two runs (identical runs differ by 4 %). Here ONE process builds model, optimiser and EMA once and times short blocks of real training steps
(forward, backward, clip, optimiser step, EMA, data wait) with the settings of one arm, then the next arm, in the order A B | B A | A B | B A ...: the two blocks of a
round are neighbours in time, so a drift of the machine cancels in the paired difference (exactly for an even number of rounds and a drift that is linear in time), and the
startup is paid once. Every block starts with the same random seed in all arms of a round (same samples, same mirror flips); the loader is rebuilt for every block
(pin_memory, workers and prefetch_factor are loader settings); the first block of the process is a priming block that is not counted, and a few warm-up steps of every
block are not counted either. A step with a non-finite loss is not timed (it is shorter) and is reported as skipped. Loader arms: a new loader's prefetch queue is full when the
counted window starts (this favours a deeper prefetch_factor / more workers), and workers=0 against workers=N draws the mirror flips in other processes, so those arms do not see
the same flips: compare them with a longer --warm (at least prefetch_factor x workers / accum steps) and read the result as indicative.

    ~/miniconda3/envs/gcar/bin/python training/ab_timing.py <composed config.json> --arm plain --arm "fast:@fast" --arm "nopin:@fast,pin_memory=false" [--rounds 4] [--json out.json]

    --arm NAME[:ITEM,ITEM,...]   ITEM is @preset (training/configs/presets/<preset>.json, flattened) or key=value (a JSON value: true / false / 2, not True); the first arm is
                                 the reference. An arm may only set what can be switched between two blocks of one process (SWITCHES): model.share_mover, model.splat_fused,
                                 model.sync_free, loss.sync_free, pin_memory, prefetch_factor, workers, cudnn_benchmark, tf32. Anything built into the network (lane_branch,
                                 camera_student ...) is a different model: time it with run.py and `python -m trainsuite compare`.
    --rounds N                   rounds of all arms, at least 2, default 4; use an EVEN number (an odd one leaves a part of a linear drift in the difference)
    --check                      print the plan and the time it needs, build nothing

Is a difference real? The per-round differences against the reference are a paired sample; the tool prints their mean and a two-sided 95 % t-interval and says "resolved" only
if the interval excludes zero. With few rounds only large differences can be resolved (3 rounds: about 3x the scatter of the per-round differences): run an A/A first (two arms
with the same settings, e.g. --arm a --arm b) to see what the noise of this box is.

The config is a composed one (`python -m trainsuite check job.json` for a job file with config t20_base.json, id ab, preset timing, set split=small and no_card prints the path of the
composed file under temp/compose/ and starts nothing; this tool reads the trainer config, not the plan). It is built the way train() builds it (speed flags, freeze, warm start from
init_from, EMA without the frozen keys); the learning rate is constant (a tenth of the config's: no warm-up to run through, no divergence of a model that is trained for a few
steps), nothing is saved. The model trains while it is timed: the numbers are step times, not results. Start it like a timing run, alone on the GPU, under a machine lock
(`tools/machine_lock.py take ... --exclusive`) and run.py's slots (`python3 ../autoresearch/tools/run.py --id ab.x --gpu --no-card -- ~/miniconda3/envs/gcar/bin/python
training/ab_timing.py ...`); it refuses when another job holds GPU memory or when the config asks for cuda and there is none. Self-test: tests/test_ab_timing.py.
"""
import argparse
import dataclasses
import gc
import json
import math
import os
import sys
import time

import torch

try:
    from . import train as T
except ImportError:                                   # run as a script
    import train as T
from trainkit.amp import make_scaler              # noqa: E402  (importing train put autoresearch/ on sys.path)
from trainkit.checkpoint import warm_start        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = os.path.join(HERE, "configs", "presets")
BOOL_SWITCHES = ("model.share_mover", "model.splat_fused", "model.sync_free", "loss.sync_free", "pin_memory", "cudnn_benchmark", "tf32")
INT_SWITCHES = ("prefetch_factor", "workers")
SWITCHES = BOOL_SWITCHES + INT_SWITCHES
LR_FRACTION = 0.1                    # the learning rate while timing: constant, a tenth of the config's
GPU_BUSY_MB = 3000                   # more GPU memory in use than this before the model is built: another job is there (a desktop holds ~1.5 GB)
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10,
          19: 2.09, 20: 2.09, 21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06, 26: 2.06, 27: 2.05, 28: 2.05, 29: 2.05, 30: 2.04}       # two-sided 95 %, by degrees of freedom; 2.0 beyond 30


# ------------------------------------------------------------------------------------------------ arms
def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(_flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def parse_arm(spec, presets=PRESETS):
    """'name' or 'name:item,item' -> (name, {switch: value}); an item is @preset or key=value (value as JSON). Only SWITCHES are allowed, with the right type: true / false for
    the switches, an integer >= 0 for workers and prefetch_factor (a string such as False would be truthy)"""
    name, _, items = spec.partition(":")
    name = name.strip()
    if not name or not all(c.isalnum() or c in "_-." for c in name):
        raise ValueError(f"arm {spec!r}: the name must be letters, digits, _ - . (got {name!r})")
    settings = {}
    for it in (i.strip() for i in items.split(",")):
        if not it:
            continue
        if it.startswith("@"):
            path = os.path.join(presets, it[1:] + ".json")
            if not os.path.isfile(path):
                raise ValueError(f"arm {name}: no preset {it[1:]!r} in {presets}")
            with open(path) as f:
                settings.update(_flatten(json.load(f)))
        elif "=" in it:
            k, _, v = it.partition("=")
            try:
                settings[k.strip()] = json.loads(v)
            except ValueError:
                raise ValueError(f"arm {name}: the value of {k.strip()} is not JSON ({v!r}); write true / false / a number")
        else:
            raise ValueError(f"arm {name}: {it!r} is neither @preset nor key=value")
    bad = sorted(set(settings) - set(SWITCHES))
    if bad:
        raise ValueError(f"arm {name}: {bad} cannot be switched between two blocks of one process; allowed: {list(SWITCHES)}")
    for k, v in settings.items():
        if k in BOOL_SWITCHES and not isinstance(v, bool):
            raise ValueError(f"arm {name}: {k} must be true or false (got {v!r})")
        if k in INT_SWITCHES and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError(f"arm {name}: {k} must be an integer >= 0 (got {v!r})")
    return name, settings


# ------------------------------------------------------------------------------------------------ the A/B schedule and its statistics (no torch inside)
def schedule(n_arms, rounds):
    """the order of the arms in every round: A B C | C B A | A B C ... (reversed on odd rounds, so a drift of the machine hits every arm alike)"""
    base = list(range(n_arms))
    return [base if r % 2 == 0 else base[::-1] for r in range(rounds)]


def paired_stats(per_round):
    """mean of the per-round differences (%), its standard error, the two-sided 95 % half-width and whether the interval excludes zero"""
    n = len(per_round)
    mean = sum(per_round) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in per_round) / (n - 1)) if n > 1 else float("inf")
    se = sd / math.sqrt(n)
    half = T_CRIT.get(n - 1, 2.0) * se
    return {"mean": mean, "se": se, "half": half, "resolved": abs(mean) > half}


def run_ab(arms, block, rounds, prime=True, on_block=None):
    """arms: [(name, settings)], the first is the reference. block(settings, r) -> {"times": [seconds per counted step], "wait": [...], optional "skipped", "load", "peak_gb"};
    r is the round (-1 for the priming block, which is run with the reference arm first and not counted). on_block(blocks) is called after every counted block with the blocks so
    far (a failure later does not lose them). -> {"arms", "ref", "diff", "blocks"}"""
    if len(arms) < 2:
        raise ValueError("an A/B needs at least two arms")
    if rounds < 2:
        raise ValueError("an A/B needs at least two rounds (a paired difference with one pair has no scatter); 4 or more is advised")
    names = [a[0] for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must differ: {names}")
    if prime:
        block(arms[0][1], -1)
    means = {n: [None] * rounds for n in names}
    waits = {n: [] for n in names}
    extra = {n: {"load": [], "peak_gb": [], "skipped": 0} for n in names}
    blocks = []
    for r, order in enumerate(schedule(len(arms), rounds)):
        for i in order:
            name, settings = arms[i]
            res = block(settings, r)
            if not res["times"]:
                raise RuntimeError(f"arm {name}, round {r}: no step was counted (every step was skipped as non-finite?)")
            means[name][r] = sum(res["times"]) / len(res["times"])
            waits[name] += res["wait"]
            extra[name]["skipped"] += res.get("skipped", 0)
            for k in ("load", "peak_gb"):
                if res.get(k) is not None:
                    extra[name][k].append(res[k])
            blocks.append({"arm": name, "round": r, "mean": means[name][r], **{k: res.get(k) for k in ("times", "wait", "skipped", "load", "peak_gb")}})
            if on_block is not None:
                on_block(blocks)
    out = {"arms": {}, "ref": names[0], "diff": {}, "blocks": blocks, "rounds": rounds}
    for n in names:
        m = means[n]
        e = extra[n]
        out["arms"][n] = {"means": m, "mean": sum(m) / len(m), "min": min(m), "max": max(m), "wait": sum(waits[n]) / max(1, len(waits[n])), "skipped": e["skipped"],
                          "load": sum(e["load"]) / len(e["load"]) if e["load"] else None, "peak_gb": max(e["peak_gb"]) if e["peak_gb"] else None}
    ref = means[names[0]]
    for n in names[1:]:
        per = [(means[n][r] / ref[r] - 1) * 100 for r in range(rounds)]
        out["diff"][n] = {"per_round": per, **paired_stats(per)}
    return out


def report(res, steps, warm):
    ref, rounds = res["ref"], res["rounds"]
    lines = [f"A/B timing: {len(res['arms'])} arms x {rounds} rounds x {steps} counted steps ({warm} warm-up steps per block and one priming block not counted); reference: {ref}",
             f"{'arm':14s} {'s/step':>7s} {'min':>6s} {'max':>6s} {'data wait':>10s} {'box load':>8s} {'peak GB':>8s}  vs {ref} (per round, %)"]
    for n, a in res["arms"].items():
        d = res["diff"].get(n)
        vs = "" if d is None else f"{d['mean']:+6.1f}   " + " ".join(f"{x:+.1f}" for x in d["per_round"])
        load = "-" if a.get("load") is None else f"{a['load']:.1f}"
        peak = "-" if a.get("peak_gb") is None else f"{a['peak_gb']:.1f}"
        lines.append(f"{n:14s} {a['mean']:7.3f} {a['min']:6.3f} {a['max']:6.3f} {1000 * a['wait']:7.0f} ms {load:>8s} {peak:>8s}  {vs}")
    for n, d in res["diff"].items():
        verdict = "RESOLVED (the 95 % interval excludes zero)" if d["resolved"] else "not resolved (the interval contains zero: no difference shown by this measurement)"
        lines.append(f"{n} against {ref}: {d['mean']:+.1f} % +- {d['half']:.1f} % (95 %, {rounds - 1} degrees of freedom); {verdict}")
    skipped = {n: a["skipped"] for n, a in res["arms"].items() if a.get("skipped")}
    if skipped:
        lines.append(f"WARNING: steps with a non-finite loss were skipped and not timed: {skipped}")
    if rounds % 2:
        lines.append("note: an odd number of rounds leaves a part of a linear drift of the machine in the difference; use an even one")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ one training, built once
class Bench:
    """model, loss, optimiser and EMA of one training; `block(settings, r)` runs `warm` + `steps` real training steps with these settings and returns the step times.
    make_loader(cfg) -> an iterable of micro-batches (clips) for the loader settings of cfg (pin_memory, workers, prefetch_factor)."""

    def __init__(self, cfg, model, loss, make_loader, device, warm=3, steps=8, frozen=()):
        self.cfg, self.model, self.loss, self.make_loader, self.device = cfg, model, loss, make_loader, device
        self.warm, self.steps = warm, steps
        self._cuda = self._sync_gpu = str(device).startswith("cuda")         # _sync_gpu: whether to wait for the GPU around a step (a test switches it separately)
        self.model.train()
        self.opt = torch.optim.AdamW(T.param_groups(model, loss, cfg.lr * LR_FRACTION, cfg.weight_decay, cfg.lr_groups), lr=cfg.lr * LR_FRACTION, betas=(0.9, 0.999))
        self.params = [p for g in self.opt.param_groups for p in g["params"]]
        self.clip_split = T.clip_groups(model, cfg.grad_clip_split)
        self.scaler = make_scaler(cfg, device)
        self.ema = T.EMA(model, cfg.ema, skip=frozen)
        self.base = self.current()
        self.n = 0

    def current(self):
        c = {}
        for k in SWITCHES:
            if k.startswith("model."):
                c[k] = getattr(self.model.cfg, k[6:])
            elif k.startswith("loss."):
                c[k] = getattr(self.loss.cfg, k[5:])
            elif k == "cudnn_benchmark":
                c[k] = bool(torch.backends.cudnn.benchmark)
            elif k == "tf32":
                c[k] = torch.get_float32_matmul_precision() == "high"
            else:
                c[k] = getattr(self.cfg, k)
        return c

    def effective(self, settings):
        """the full switch set an arm runs with: what it names, the start value for everything else"""
        return {**self.base, **settings}

    def apply(self, settings):
        """every switch gets the arm's value or, if the arm does not name it, the value it had at the start (an arm never inherits from the arm before it) -> the TrainConfig of the block"""
        full = self.effective(settings)
        for k, v in full.items():
            if k.startswith("model."):
                setattr(self.model.cfg, k[6:], v)
            elif k.startswith("loss."):
                setattr(self.loss.cfg, k[5:], v)
            elif k == "cudnn_benchmark":
                torch.backends.cudnn.benchmark = bool(v)
            elif k == "tf32":
                torch.set_float32_matmul_precision("high" if v else "highest")
        return dataclasses.replace(self.cfg, **{k: full[k] for k in ("pin_memory", "prefetch_factor", "workers")})

    def _sync(self):
        if self._sync_gpu:
            torch.cuda.synchronize()

    def block(self, settings, r=0):
        cfg = self.apply(settings)
        torch.manual_seed(cfg.seed + 7919 * (r + 1))            # the same samples and mirror flips in every arm of a round
        cuda = self._cuda
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        loader = self.make_loader(cfg)
        it = iter(loader)
        times, waits, skipped = [], [], 0
        for i in range(self.warm + self.steps):
            self._sync()
            t0 = time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            wait, finite = 0.0, True
            for _ in range(cfg.accum):
                t1 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    it = iter(loader)
                    item = next(it)
                wait += time.perf_counter() - t1
                out, ok, _, _ = T.run_clip(self.model, self.loss, item if isinstance(item, list) else [item], 0.5, self.device, cfg.amp, cfg,
                                           diag=False, backward=True, augment=True, scaler=self.scaler)
                if not ok:
                    finite = False
                    break
            if finite:
                _, status = T.optimizer_step(self.opt, self.params, cfg.grad_clip, self.scaler, split=self.clip_split)
            else:
                self.opt.zero_grad(set_to_none=True)
                status = "nonfinite"
            if status == "ok":
                self.ema.update(self.model, self.n)
            self.n += 1
            self._sync()
            if i >= self.warm:
                if status == "ok":
                    times.append(time.perf_counter() - t0)
                    waits.append(wait)
                else:
                    skipped += 1                                # a skipped step is shorter (no optimiser step, no EMA): not timed
        del it, loader                                          # the workers of this block end here
        gc.collect()
        return {"times": times, "wait": waits, "skipped": skipped, "load": os.getloadavg()[0], "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None}


# ------------------------------------------------------------------------------------------------ command line
def gpu_used_mb():
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 20


def build_bench(cfg, warm, steps):
    """the real training of a composed config, the way train() builds it (speed flags, model, loss, freeze, datasets, clips, warm start, EMA), without the loop, the checkpoints and the logs"""
    T.seed_all(cfg.seed)
    device = T.resolve_device(cfg.device)
    if str(cfg.device).startswith("cuda") and not str(device).startswith("cuda"):
        raise SystemExit(f"refused: the config asks for {cfg.device} and there is no CUDA device (resolve_device gave {device}); the step times would be those of the CPU")
    done = T.apply_speed_flags(cfg)
    if done:
        print("speed flags of the config:", done, flush=True)
    model, loss, _ = T.build(cfg)
    model, loss = model.to(device), loss.to(device)
    frozen = ()
    if cfg.train_only or cfg.freeze:
        T.freeze_mod.apply(model, cfg.train_only, cfg.freeze)
        frozen = T.freeze_mod.frozen_keys(model)
    if cfg.init_from:
        warm_start(cfg.init_from, model, loss, reset=cfg.init_reset)
    train_ds, val_ds = T.make_datasets(cfg)
    T.check_grids(model, train_ds, val_ds)
    if cfg.seq_len > 1:
        train_ds = T.ClipDataset(train_ds, cfg.seq_len)
    return Bench(cfg, model, loss, lambda c: T.make_loader(train_ds, c, True), device, warm, steps, frozen=frozen)


def main(argv=None, builder=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a composed config (trainsuite check writes it under temp/compose/)")
    ap.add_argument("--arm", action="append", required=True, help="NAME[:@preset,key=value,...]; the first is the reference; at least two")
    ap.add_argument("--rounds", type=int, default=4, help="rounds of all arms (the order alternates), at least 2, an even number; default 4")
    ap.add_argument("--steps", type=int, default=8, help="counted steps per block; default 8")
    ap.add_argument("--warm", type=int, default=3, help="warm-up steps per block, not counted; default 3")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--check", action="store_true", help="print the plan and stop")
    ap.add_argument("--force", action="store_true", help=f"run although more than {GPU_BUSY_MB} MB of GPU memory are in use")
    a = ap.parse_args(argv)
    try:
        arms = [parse_arm(s) for s in a.arm]
    except ValueError as e:
        ap.error(str(e))
    if len(arms) < 2:
        ap.error("an A/B needs at least two --arm")
    if len({n for n, _ in arms}) != len(arms):
        ap.error(f"arm names must differ: {[n for n, _ in arms]}")
    if a.rounds < 2 or a.steps < 1 or a.warm < 0:
        ap.error("--rounds must be >= 2, --steps >= 1, --warm >= 0")
    blocks = a.rounds * len(arms) + 1
    print(f"plan: {len(arms)} arms x {a.rounds} rounds = {blocks - 1} blocks of {a.warm} + {a.steps} steps and one priming block; order per round " +
          " | ".join(" ".join(arms[i][0] for i in o) for o in schedule(len(arms), a.rounds)[:2]) + " ...")
    for n, s in arms:
        print(f"  {n:14s} {s if s else '(the config as it is)'}")
    print(f"  about {blocks * (a.warm + a.steps) * 3.5 / 60 + 1.5:.0f} min (a guess: 3.5 s per step plus the start)")
    if a.check:
        return 0
    used = gpu_used_mb()
    print(f"GPU memory in use before the model: {used:.0f} MB")
    if used > GPU_BUSY_MB and not a.force:
        print(f"refused: another job holds GPU memory (> {GPU_BUSY_MB} MB); the step times would not be clean (--force to run anyway)")
        return 2
    bench = (builder or build_bench)(T.read_config(a.config), a.warm, a.steps)
    info = dict(steps=a.steps, warm=a.warm, config=os.path.abspath(a.config), settings={n: bench.effective(s) for n, s in arms})

    def write(res):
        if a.json:
            with open(a.json, "w") as f:
                json.dump({**res, **info}, f, indent=1)
    try:
        res = run_ab(arms, bench.block, a.rounds, on_block=lambda blocks: write({"partial": True, "blocks": blocks}))      # the file holds the blocks so far if a later one fails
    finally:
        bench.apply({})                                         # model.cfg, loss.cfg and the torch switches go back to what they were at the start
    print(report({**res, **info}, a.steps, a.warm))
    for n, s in info["settings"].items():
        print(f"  effective settings of {n}: {s}")
    write(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

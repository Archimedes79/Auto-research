"""The in-process A/B timer of the speed switches (training/ab_timing.py). CPU, no data, about a minute.

  1. parse_arm: names, @preset (flattened), key=value (JSON values), and the refusals: a setting that cannot be switched between two blocks of a process, a value of the wrong type
     (Python's False is not JSON and a string would be truthy), a bad preset, a bad item, a bad name
  2. schedule: A B | B A | A B ... (reversed on odd rounds), also for three arms
  3. run_ab with fake blocks of known step times and a drift of the machine: the priming block comes first and is not counted, every arm gets `rounds` blocks in the schedule's order,
     the drift cancels in the paired difference, the block mean is the mean of its steps, load / peak / skipped are aggregated, paired_stats on known numbers (a clear effect is
     resolved, a scatter around zero is not), the report says so, the refusals (one arm, equal names, one round, a block without counted steps)
  4. Bench.apply / current / effective: every switch gets the arm's value or the start value (an arm does not inherit from the arm before), the loader config carries pin_memory /
     workers / prefetch_factor, the global torch switches (cudnn benchmark, tf32) go both ways
  5. real training steps of a tiny model on the CPU, with spies on what a step is made of: accum micro-batches per step each timed as data wait, run_clip gets the arm's config,
     the clip split and the clip norm reach the optimiser step, the EMA is updated once per step, the gradients are zeroed, the learning rate is a tenth of the config's, the same
     seed opens every arm of a round and differs between rounds, a loader that runs out starts again, the loader of a block is freed, a step with a non-finite loss is not timed
     but counted, warm-up steps are not counted, the GPU is waited for before and after each step
  6. the command line end to end with a builder that makes the tiny bench: the result file (blocks, effective settings, steps), the verdict text, --check builds nothing, one arm /
     equal names / one round are refused, a busy GPU is refused and --force runs, a config that asks for cuda without a GPU is refused, gpu_used_mb reads total - free
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_ab_timing.py   (PYTHONPATH = the repository root for autogt)"""
import contextlib
import copy
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import time
import weakref
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, "..")]
import ab_timing as AB  # noqa: E402
import losses as L  # noqa: E402
import synthetic_world as sw  # noqa: E402
import train as T  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ---- 1 parse_arm
n, s = AB.parse_arm("plain")
check(n == "plain" and s == {}, "parse_arm: a bare name is an arm with the config as it is")
n, s = AB.parse_arm("fast:@fast")
check(n == "fast" and s == {"model.share_mover": True, "model.splat_fused": True, "model.sync_free": True, "loss.sync_free": True, "pin_memory": True}, f"parse_arm: @fast is the flattened preset ({s})")
n, s = AB.parse_arm("nopin:@fast,pin_memory=false,workers=2,prefetch_factor=4")
check(s["pin_memory"] is False and s["workers"] == 2 and s["prefetch_factor"] == 4 and s["model.sync_free"] is True, "parse_arm: key=value after a preset overrides it, values are JSON (false, 2, 4)")
check(AB.parse_arm("x:tf32=true,cudnn_benchmark=true")[1] == {"tf32": True, "cudnn_benchmark": True}, "parse_arm: the global torch switches")
for bad, why in (("a:model.lane_branch=\"res4\"", "a setting built into the network"), ("a:@timing", "a preset with steps"), ("a:@nope", "an unknown preset"), ("a:foo", "an item that is neither"),
                 ("a b", "a name with a space"), (":pin_memory=true", "an empty name"), ("a:accum=2", "a setting that changes the work of a step"),
                 ("a:pin_memory=False", "Python's False (not JSON: it would become a truthy string)"), ("a:model.sync_free=no", "a string for a switch"), ("a:pin_memory=0", "0 for a switch"),
                 ("a:workers=true", "true for workers"), ("a:workers=-1", "a negative worker count"), ("a:workers=2.5", "a fractional worker count"), ("a:prefetch_factor=\"x\"", "a string for prefetch_factor")):
    check(raises(lambda b=bad: AB.parse_arm(b)), f"parse_arm refuses {why} ({bad!r})")
msgs = {}
for bad in ('a:model.lane_branch="res4"', "a:pin_memory=False", "a:workers=-1", "a:model.sync_free=\"no\""):
    try:
        AB.parse_arm(bad)
    except ValueError as e:
        msgs[bad] = str(e)
check("cannot be switched" in msgs.get('a:model.lane_branch="res4"', "") and "not JSON" in msgs.get("a:pin_memory=False", "") and ">= 0" in msgs.get("a:workers=-1", "")
      and "true or false" in msgs.get('a:model.sync_free="no"', ""), "parse_arm: each refusal says why (not switchable / not JSON / integer >= 0 / true or false)")

# ---- 2 schedule
check(AB.schedule(2, 4) == [[0, 1], [1, 0], [0, 1], [1, 0]] and AB.schedule(3, 3) == [[0, 1, 2], [2, 1, 0], [0, 1, 2]], "schedule: A B | B A | A B ..., three arms C B A on odd rounds")

# ---- 3 run_ab and the statistics
calls = []
state = {"n": 0}
TIMES = {"a": 1.0, "b": 0.9, "c": 1.1}


def fake_block(settings, r):
    drift = 1 + 0.01 * state["n"]                                       # the machine slows down by 1 % per block
    state["n"] += 1
    calls.append((settings["arm"], r))
    t = TIMES[settings["arm"]] * drift
    return {"times": [t, t], "wait": [0.01, 0.01], "load": 2.0, "peak_gb": 7.5, "skipped": 0}


arms = [("a", {"arm": "a"}), ("b", {"arm": "b"})]
res = AB.run_ab(arms, fake_block, 4)
check(calls == [("a", -1)] + [(x, r) for r, o in enumerate([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")]) for x in o], f"run_ab: a priming block of the reference arm (round -1) first, then the schedule ({calls[:4]} ...)")
state["n"], calls[:] = 0, []
res = AB.run_ab(arms, fake_block, 4, prime=False)
check(len(res["blocks"]) == 8 and calls[0] == ("a", 0) and all(len(res["arms"][k]["means"]) == 4 for k in "ab"), "run_ab: eight counted blocks, none for the priming block (prime=False: none run)")
check(abs(res["diff"]["b"]["mean"] + 10.0) < 0.3 and all(x < 0 for x in res["diff"]["b"]["per_round"]) and res["diff"]["b"]["resolved"],
      f"run_ab: b is 10 % faster than a, the drift of 1 % per block cancels in the pairs, the difference is resolved ({res['diff']['b']['mean']:+.2f} %)")
check(abs(res["arms"]["a"]["wait"] - 0.01) < 1e-12 and res["ref"] == "a" and res["arms"]["a"]["load"] == 2.0 and res["arms"]["a"]["peak_gb"] == 7.5 and res["arms"]["a"]["skipped"] == 0,
      "run_ab: data wait averaged, load averaged, peak memory is the largest, the first arm is the reference")
r2 = AB.run_ab(arms, lambda s, r: {"times": [1.0, 3.0] if s["arm"] == "a" else [1.0, 1.0], "wait": [0.0, 0.0], "skipped": 2}, 2, prime=False)
check(r2["arms"]["a"]["mean"] == 2.0 and r2["arms"]["b"]["mean"] == 1.0 and r2["arms"]["a"]["skipped"] == 4 and r2["arms"]["a"]["load"] is None, "run_ab: the block mean is the mean of its steps (not the min or the last), skipped steps are summed")
state["n"], calls[:] = 0, []
res3 = AB.run_ab([("a", {"arm": "a"}), ("b", {"arm": "b"}), ("c", {"arm": "c"})], fake_block, 2, prime=False)
check([c[0] for c in calls] == ["a", "b", "c", "c", "b", "a"] and res3["diff"]["c"]["mean"] > 0 > res3["diff"]["b"]["mean"], "run_ab: three arms, C B A on the second round, c slower and b faster than a")
check(raises(lambda: AB.run_ab([("a", {})], fake_block, 2)) and raises(lambda: AB.run_ab([("a", {}), ("a", {})], fake_block, 2)) and raises(lambda: AB.run_ab(arms, fake_block, 1))
      and raises(lambda: AB.run_ab(arms, lambda s, r: {"times": [], "wait": []}, 2, prime=False), RuntimeError), "run_ab refuses one arm, equal names, one round, and a block without counted steps")
st = AB.paired_stats([-5.0, -5.2, -4.8, -5.0])
st2 = AB.paired_stats([3.0, -4.0, 5.0, -3.0])
st3 = AB.paired_stats([-4.0, -6.0])
check(st["resolved"] and abs(st["mean"] + 5.0) < 1e-9 and 0 < st["half"] < 1.0 and not st2["resolved"] and st2["half"] > abs(st2["mean"]) and not st3["resolved"] and abs(st3["half"] - 12.71) < 1e-6,
      f"paired_stats: [-5.0 -5.2 -4.8 -5.0] resolved (+-{st['half']:.2f}), a scatter around zero is not, two rounds need the t of one degree of freedom (12.7)")
check(not AB.paired_stats([2.0, 2.0, 2.0, 2.0])["half"] > 0 and AB.paired_stats([2.0, 2.0, 2.0, 2.0])["resolved"], "paired_stats: identical per-round differences have no scatter")
txt = AB.report(AB.run_ab(arms, fake_block, 4, prime=False), 2, 1)
check("RESOLVED" in txt and "against a" in txt and "-10." in txt and "odd number" not in txt and "WARNING" not in txt, "report: a clear difference says RESOLVED with its interval; no odd-rounds note, no warning")
noisy = iter([1.0, 1.0, 1.0, 1.0 + 0.2, 1.0, 1.0 - 0.2, 1.0, 1.0 + 0.1])
txt_noisy = AB.report(AB.run_ab([("a", {"arm": "a"}), ("b", {"arm": "b"})], lambda s, r: {"times": [next(noisy)], "wait": [0.0]}, 4, prime=False), 1, 0)
check("not resolved" in txt_noisy and "RESOLVED" not in txt_noisy.replace("not resolved", ""), "report: arms that differ by scatter only are 'not resolved'")
txt3 = AB.report(AB.run_ab(arms, lambda s, r: {"times": [1.0], "wait": [0.0], "skipped": 1}, 3, prime=False), 1, 0)
check("odd number" in txt3 and "WARNING" in txt3 and "skipped" in txt3, "report: an odd number of rounds and skipped steps are named")

# ---- 4 apply / current / effective
mcfg = SimpleNamespace(share_mover=False, splat_fused=False, sync_free=False)
lcfg = SimpleNamespace(sync_free=False)
bench = AB.Bench.__new__(AB.Bench)
bench.cfg, bench.model, bench.loss = T.TrainConfig(), SimpleNamespace(cfg=mcfg), SimpleNamespace(cfg=lcfg)
bench.base = bench.current()
tf0, cb0 = torch.get_float32_matmul_precision(), torch.backends.cudnn.benchmark
try:
    c1 = bench.apply({"model.sync_free": True, "loss.sync_free": True, "pin_memory": True, "workers": 3, "tf32": True, "cudnn_benchmark": True})
    check(mcfg.sync_free and lcfg.sync_free and not mcfg.share_mover and c1.pin_memory and c1.workers == 3 and torch.get_float32_matmul_precision() == "high" and torch.backends.cudnn.benchmark,
          "apply: the arm's switches are set (model, loss, loader config, tf32, cudnn benchmark)")
    c2 = bench.apply({"model.share_mover": True})
    check(mcfg.share_mover and not mcfg.sync_free and not lcfg.sync_free and not c2.pin_memory and c2.workers == bench.base["workers"] and torch.get_float32_matmul_precision() == tf0
          and torch.backends.cudnn.benchmark == cb0, "apply: the next arm does not inherit from the one before (everything it does not name is back at the start value)")
    check(bench.current() == {**bench.base, "model.share_mover": True} == bench.effective({"model.share_mover": True}), "current / effective: read back what apply set")
finally:
    torch.set_float32_matmul_precision(tf0)
    torch.backends.cudnn.benchmark = cb0

# ---- 5 real steps of a tiny model on the CPU
Bn = 2
mcfg_real = ModelConfig(**dict(BASE, camera_features="dual", cam_temporal="window_memory"))
torch.manual_seed(0)
m = CachedBEVFusionModel(mcfg_real)
with torch.no_grad():
    for q in m.parameters():
        if q.requires_grad and q.abs().max() == 0:
            q.normal_(0, 1e-2)
loss = L.BEVLoss(grid=m.grid)
scene = sw.Scene([sw.Box(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), sw.Box(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0)], lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
side = {"prev": L.synthetic_gt(scene.step(-0.5), m.grid), "cur": L.synthetic_gt(scene, m.grid), "next": L.synthetic_gt(scene.step(0.5), m.grid)}
item = dict(batch_for(m.cfg, B=Bn), gt=[side] * Bn, tokens=["a", "b"], gap_steps=[{"prev": [(-0.5, 8.0, 0.05)], "next": [(0.5, 8.0, 0.05)]}] * Bn)
tcfg = T.TrainConfig(accum=2, batch_size=Bn, seq_len=1, flip_aug=0.5, device="cpu", workers=0, grad_clip=5.0, grad_clip_split=("cam_reduce",), lr=1e-3, seed=3)
made = []


class Loader:
    """an iterable of fresh copies of the batch; counts the batches taken, sleeps a little per batch (the data wait), can run out"""
    taken = 0

    def __init__(self, n, delay=0.0):
        self.n, self.delay = n, delay

    def __iter__(self):
        for _ in range(self.n):
            time.sleep(self.delay)
            Loader.taken += 1
            yield copy.deepcopy(item)


def make_loader(c, n=3, delay=0.0):
    made.append({"pin": c.pin_memory, "sync": m.cfg.sync_free, "loss_sync": loss.cfg.sync_free, "share": m.cfg.share_mover, "rand": float(torch.rand(1))})
    ld = Loader(n, delay)
    made[-1]["ref"] = weakref.ref(ld)
    return ld


bench = AB.Bench(tcfg, m, loss, make_loader, "cpu", warm=1, steps=2)
spy = {"run_clip": [], "opt_step": [], "ema": 0, "zero": 0, "sync": 0, "fail_at": None}
_rc, _os, _eu, _zg = T.run_clip, T.optimizer_step, bench.ema.update, bench.opt.zero_grad


def rc(*a, **k):
    spy["run_clip"].append(a[6].pin_memory)                             # the cfg run_clip got
    if spy["fail_at"] is not None and len(spy["run_clip"]) == spy["fail_at"]:
        return {"total": float("nan")}, False, None, []
    return _rc(*a, **k)


def ostep(*a, **k):
    spy["opt_step"].append((a[2], k.get("split")))
    return _os(*a, **k)


T.run_clip, T.optimizer_step = rc, ostep
bench.ema.update = lambda *a, **k: (spy.__setitem__("ema", spy["ema"] + 1), _eu(*a, **k))[1]
bench.opt.zero_grad = lambda *a, **k: (spy.__setitem__("zero", spy["zero"] + 1), _zg(*a, **k))[1]
bench._sync = lambda: spy.__setitem__("sync", spy["sync"] + 1)
before = {k: v.detach().clone() for k, v in m.named_parameters()}        # parameters, not buffers: the batch-norm statistics move in any forward
try:
    Loader.taken = 0
    blk = bench.block({"pin_memory": True}, 0)
    check(len(blk["times"]) == 2 and len(blk["wait"]) == 2 and blk["skipped"] == 0 and all(t > 0 and math.isfinite(t) for t in blk["times"]) and blk["load"] >= 0 and blk["peak_gb"] is None, "Bench.block: 2 counted steps of 1 + 2, positive finite times, none skipped, the box load recorded (no peak memory on the CPU)")
    check(Loader.taken == 3 * tcfg.accum and len(spy["run_clip"]) == 3 * tcfg.accum, f"a step is accum = {tcfg.accum} micro-batches ({Loader.taken} batches for 3 steps)")
    check(spy["run_clip"] == [True] * 6, "run_clip gets the arm's config (pin_memory True from the arm), not the base config")
    check(spy["opt_step"] == [(tcfg.grad_clip, bench.clip_split)] * 3 and bench.clip_split, "the optimiser step gets the clip norm and the clip split of the config")
    check(spy["ema"] == 3 and spy["zero"] >= 3, f"the EMA is updated once per step ({spy['ema']} of 3) and the gradients are zeroed ({spy['zero']})")
    check(len(spy["run_clip"]) == 6 and abs(bench.opt.param_groups[0]["lr"] - tcfg.lr * AB.LR_FRACTION) < 1e-12, f"the learning rate is {AB.LR_FRACTION} of the config's")
    check(spy["sync"] == 6, f"the GPU is waited for before the timer starts and after each step ({spy['sync']} for 3 steps)")
    moved = [k for k, v in m.named_parameters() if not torch.equal(before[k], v.detach())]
    check(len(moved) > 10, f"the optimiser really stepped ({len(moved)} parameters moved)")
    # the same seed opens every arm of a round, another one the next round
    made.clear()
    for settings, r in (({}, 1), ({"model.sync_free": True}, 1), ({}, 2)):
        bench.block(settings, r)
    check(made[0]["rand"] == made[1]["rand"] != made[2]["rand"] and made[0]["rand"] == (torch.manual_seed(tcfg.seed + 7919 * 2), float(torch.rand(1)))[1],
          "the same seed opens both arms of a round (the same samples and flips), another opens the next round")
    check([(d["sync"], d["loss_sync"]) for d in made[:2]] == [(False, False), (True, False)], "the flags of the arm are set when its loader is built, the one before is not inherited")
    # a loader that runs out starts again
    made.clear()
    bench.make_loader = lambda c: make_loader(c, n=1)
    bench.block({}, 3)
    check(Loader.taken > 0 and len(made) == 1, "a loader with fewer batches than the block needs starts again (no StopIteration)")
    # the loader of a block is freed
    check(made[0]["ref"]() is None, "the loader of a block is freed when the block ends (its workers end with it)")
    # the data wait is measured
    bench.make_loader = lambda c: make_loader(c, n=100, delay=0.05)
    bw = bench.block({}, 4)
    check(all(w >= 0.05 * tcfg.accum * 0.9 for w in bw["wait"]), f"the data wait is what the loader took ({min(bw['wait']):.3f} s per step for 2 x 0.05 s)")
    # a step with a non-finite loss is not timed but counted
    bench.make_loader = make_loader
    spy["run_clip"].clear()
    spy["fail_at"] = 5                                                  # the first micro-batch of the third step (the second counted one)
    bs = bench.block({}, 5)
    check(bs["skipped"] == 1 and len(bs["times"]) == 1, f"a step with a non-finite loss is not timed and counted as skipped ({bs['skipped']} skipped, {len(bs['times'])} timed)")
finally:
    T.run_clip, T.optimizer_step = _rc, _os
    bench.ema.update, bench.opt.zero_grad = _eu, _zg
    del bench._sync

# ---- 6 the command line end to end
cfg_json = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(dataclasses.asdict(tcfg), cfg_json)
cfg_json.close()
out_json = cfg_json.name + ".out.json"
built = []


def builder(cfg, warm, steps):
    built.append((cfg.seed, warm, steps))
    return AB.Bench(cfg, m, loss, lambda c: make_loader(c), "cpu", warm, steps)


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_ = AB.main([cfg_json.name, "--arm", "plain", "--arm", "fast:model.sync_free=true,loss.sync_free=true", "--rounds", "2", "--steps", "2", "--warm", "1", "--json", out_json], builder=builder)
text = buf.getvalue()
res_file = json.load(open(out_json)) if os.path.exists(out_json) else {}
check(rc_ == 0 and built == [(3, 1, 2)] and "plan: 2 arms x 2 rounds" in text and "A/B timing:" in text and ("RESOLVED" in text or "not resolved" in text), "main: reads the config, builds the bench with warm and steps, prints plan and report with a verdict")
check(len(res_file.get("blocks", [])) == 4 and res_file.get("steps") == 2 and res_file.get("warm") == 1 and set(res_file.get("settings", {})) == {"plain", "fast"}
      and res_file["settings"]["fast"]["model.sync_free"] is True and res_file["settings"]["plain"]["model.sync_free"] is False and all(len(b["times"]) == 2 and b["load"] is not None for b in res_file["blocks"]),
      "main --json: four blocks with their step times, steps and warm, the effective settings of both arms")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_ = AB.main(["nowhere.json", "--arm", "plain", "--arm", "fast:@fast", "--rounds", "2", "--check"])
out = buf.getvalue()
check(rc_ == 0 and "plan: 2 arms x 2 rounds" in out and "model.sync_free" in out and "plain fast | fast plain" in out, "main --check: prints the plan (arms, settings, order) and builds nothing (the config does not exist)")
with contextlib.redirect_stderr(io.StringIO()) as err:
    errs = [raises(lambda a=a: AB.main(a), SystemExit) for a in (["x.json", "--arm", "only", "--check"], ["x.json", "--arm", "a", "--arm", "a", "--check"], ["x.json", "--arm", "a", "--arm", "b", "--rounds", "1", "--check"],
                                                                   ["x.json", "--arm", "a", "--arm", "b:model.lane_branch=\"res4\"", "--check"], ["x.json", "--arm", "a", "--arm", "b:pin_memory=False", "--check"])]
check(all(errs) and "cannot be switched" in err.getvalue() and "must differ" in err.getvalue(), "main: one arm, equal names, one round, a setting built into the network and a string value are refused with the reason")
_gu = AB.gpu_used_mb
AB.gpu_used_mb = lambda: 5000.0
built.clear()
try:
    with contextlib.redirect_stdout(io.StringIO()):
        busy = AB.main([cfg_json.name, "--arm", "a", "--arm", "b:pin_memory=true"], builder=builder)
        forced = AB.main([cfg_json.name, "--arm", "a", "--arm", "b:model.sync_free=true", "--rounds", "2", "--steps", "1", "--warm", "1", "--force"], builder=builder)
finally:
    AB.gpu_used_mb = _gu
check(busy == 2 and forced == 0 and len(built) == 1, "main: a GPU that is busy (5000 MB in use) is refused before the model is built; --force runs")
_ic, _mf, _re = torch.cuda.is_available, getattr(torch.cuda, "mem_get_info", None), T.resolve_device
torch.cuda.is_available, torch.cuda.mem_get_info = (lambda: True), (lambda: (6 * 2 ** 30, 8 * 2 ** 30))
try:
    used = AB.gpu_used_mb()
finally:
    torch.cuda.is_available, torch.cuda.mem_get_info = _ic, _mf
check(abs(used - 2048.0) < 1e-6, f"gpu_used_mb is total - free ({used:.0f} MB for 8 GiB total, 6 GiB free)")
T.resolve_device = lambda d: "cpu"
try:
    refused = raises(lambda: AB.build_bench(dataclasses.replace(tcfg, device="cuda"), 1, 1), SystemExit)
finally:
    T.resolve_device = _re
check(refused, "build_bench: a config that asks for cuda and gets the CPU is refused (the step times would be those of the CPU)")
os.unlink(cfg_json.name)
if os.path.exists(out_json):
    os.unlink(out_json)

print("AB TIMING GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)}): " + "; ".join(fails[:5])))
sys.exit(1 if fails else 0)

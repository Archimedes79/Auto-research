"""Three guards of training/train.py found by the review of the training stack (2026-11-10; D73 lifted the freeze, each change needs a test that fails without it). CPU, toy data,
about 30 s:

  1. a stop + resume of the REAL train() is the uninterrupted run: same micro-batches in the same order (across a pass boundary), same random ops, same weights, bit for bit. The
     training loader of `make_loader` is a trainkit EpochLoader (order = f(seed, epoch), position saved in the checkpoint); a loader with an explicit generator and the validation loader
     stay plain DataLoaders
  2. evaluate(): a batch with a non-finite loss is counted (val.n, val.n_skipped), not dropped without a word; none finite gives NaN totals (not 0, not missing); with finite batches the
     numbers are the old ones bit for bit
  3. check_engine(): a pinned trainkit next to train.py that is not the one imported is refused; without a pinned copy the log says which trainkit runs (also end to end, as a script)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_train_guards.py
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
import toy_train as TT  # noqa: E402
import train as T  # noqa: E402
from trainkit.data import EpochLoader  # noqa: E402

BF = os.path.abspath(".")
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------------------------------------ 1 resume
SEEN = []                                                         # the sample of every training micro-batch, in the order the loop took them


def noisy_forward(model, loss, batch, step_frac, *a, **k):
    out, res, extra = TT.toy_forward(model, loss, batch, step_frac, *a, **k)
    if k.get("train_terms", True):
        SEEN.append(batch["tokens"][0])
    out["total"] = out["total"] + 0.01 * torch.rand(())          # a random op on the training path (the global stream), like the modality dropout
    return out, res, extra


def train_toy(root, name, stop_after=None):
    """T.train() with the real make_loader over the 40 prebuilt batches of the toy (an item is a batch, batch_size 1), 24 steps of 2 micro-batches: one pass is 40 micro-batches, so the
    run crosses a pass boundary. The stop request arrives after `stop_after` steps -> the exit code, else None"""
    keys = ("build", "forward_loss", "val_subset", "collate", "_stop_requested")
    saved, calls = {k: getattr(T, k) for k in keys}, [0]

    def stop(run_dir):
        calls[0] += 1
        return "test" if stop_after is not None and calls[0] > stop_after else None
    T.build, T.forward_loss, T.val_subset, T.collate, T._stop_requested = (lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)), noisy_forward, (lambda ds, n: ds), (lambda items: items[0]), stop
    cfg = T.TrainConfig(name=name, out_root=root, steps=24, accum=2, warmup=2, log_every=4, eval_every=0, save_every=6, keep=3, device="cpu", workers=0, batch_size=1,
                        flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5, max_bad_steps=3)
    data = TT.toy_batches()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T.train(cfg, datasets=(data, data))
    except SystemExit as e:
        return e.code
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    return None


def newest(root, name):
    d = os.path.join(root, name)
    ck = torch.load(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".pt"))[-1]), map_location="cpu", weights_only=False)
    return ck, [json.loads(ln)["total"] for ln in open(os.path.join(d, "log.jsonl"))]


ds, cfg = list(range(8)), T.TrainConfig(workers=0, batch_size=2, seed=3)
check(isinstance(T.make_loader(ds, cfg, True), EpochLoader), "make_loader: the training loader is an EpochLoader")
check(type(T.make_loader(ds, cfg, False)) is torch.utils.data.DataLoader and type(T.make_loader(ds, cfg, True, generator=torch.Generator())) is torch.utils.data.DataLoader,
      "make_loader: the validation loader and a training loader with a generator of the caller (train_motion_block) stay plain DataLoaders")
saved_collate, T.collate = T.collate, (lambda items: torch.tensor(items))                 # the items of this check are ints
try:
    order = lambda loader: [int(x) for b in loader for x in b]                                  # noqa: E731
    cfg4 = T.TrainConfig(workers=0, batch_size=2, seed=4)
    check(order(T.make_loader(ds, cfg, True)) == order(EpochLoader(ds, 2, 3, drop_last=True)) and order(T.make_loader(ds, cfg, True)) != order(T.make_loader(ds, cfg4, True)),
          "make_loader: the order of a pass is a function of cfg.seed (seed 3 is the EpochLoader of seed 3, seed 4 is another order)")
    odd = list(range(7))
    check(T.make_loader(odd, cfg, True).drop_last and len(list(T.make_loader(odd, cfg, True))) == 3 and not T.make_loader(odd, cfg, False).drop_last,
          "make_loader: the training loader drops the last incomplete batch (7 samples, batch size 2: 3 batches), the validation loader does not")
finally:
    T.collate = saved_collate
with tempfile.TemporaryDirectory() as root:
    del SEEN[:]
    check(train_toy(root, "ref") is None, "uninterrupted run: 24 steps")
    ref_seen, ref_ck, ref_totals = list(SEEN), *newest(root, "ref")
    check(len(ref_seen) == 48 and sorted(ref_seen[:40]) == sorted(f"t{i}" for i in range(40)) and ref_seen[:40] != [f"t{i}" for i in range(40)] and ref_seen[40:] != ref_seen[:8],
          "uninterrupted run: the first pass takes the 40 batches in a shuffled order, the second pass starts with other ones")
    del SEEN[:]
    check(train_toy(root, "stp", stop_after=12) == 75, "second run: stop request after step 12 (24 micro-batches, inside pass 0): exit 75")
    check(train_toy(root, "stp") is None, "second run: resumed from step 12 to 24")
    res_ck, res_totals = newest(root, "stp")
    check(SEEN == ref_seen, "stop + resume takes the same micro-batches in the same order as the uninterrupted run (across the pass boundary at micro-batch 40)")
    check(res_totals == ref_totals and TT.same_tree(ref_ck["state"], res_ck["state"]) and TT.same_tree(ref_ck["ema"], res_ck["ema"]),
          "stop + resume ends in the same logged losses, weights and EMA as the uninterrupted run, bit for bit")


# ------------------------------------------------------------------------------------------------ 2 evaluate
def eval_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    return {"total": batch["v"], "group.a": batch["v"] * 2}, {"state": {}}, None


def evaluate(values):
    """T.evaluate over one batch per value, whose loss is that value -> (result, printed text)"""
    real, model = T.forward_loss, nn.Linear(1, 1)
    T.forward_loss, buf = eval_forward, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = T.evaluate(model, nn.Module(), T.EMA(model, 0.9), [{"v": torch.tensor(v)} for v in values], 10 ** 9, "cpu", "off")
    finally:
        T.forward_loss = real
    return res, buf.getvalue()


good = [0.1, 0.2, 0.3, 0.7]
mean = lambda vs: sum(float(torch.tensor(v)) for v in vs) / len(vs)             # noqa: E731 - the old rule: the sum in loader order, divided by the number of batches
res, out = evaluate(good)
check(res == {"total": mean(good), "group.a": 2 * mean(good), "n": 4, "n_skipped": 0} and out == "", "evaluate, all finite: the old mean bit for bit, n 4, n_skipped 0, nothing printed")
res, out = evaluate([0.1, float("nan"), 0.3, float("inf")])
check(res["total"] == mean([0.1, 0.3]) and res["group.a"] == 2 * mean([0.1, 0.3]) and res["n"] == 2 and res["n_skipped"] == 2 and "left out 2 of 4" in out,
      "evaluate, 2 of 4 batches non-finite: the mean of the other two, n 2, n_skipped 2, a warning names the count")
res, out = evaluate([float("nan")] * 3)
check(math.isnan(res.get("total", 0.0)) and res["n"] == 0 and res["n_skipped"] == 3 and "left out 3 of 3" in out, "evaluate, no batch finite: total is NaN (not 0, not missing), n 0, n_skipped 3")
check(math.isnan(json.loads(json.dumps({"step": 1, "val": res}))["val"].get("total", 0.0)), "evaluate: the NaN row survives log.jsonl (json writes and reads NaN)")

# ------------------------------------------------------------------------------------------------ 3 the pinned engine
with tempfile.TemporaryDirectory() as d:
    code, live, none = (os.path.join(d, x) for x in ("code", "live", "none"))
    for p in (os.path.join(code, "trainkit"), live, none):
        os.makedirs(p)
    try:
        T.check_engine(here=code, engine=live)
        refused = None
    except SystemExit as e:
        refused = str(e.code)
    check(refused is not None and os.path.join(code, "trainkit") in refused and live in refused, "check_engine: a pinned copy next to train.py but another trainkit imported is refused, the message names both")
    line = T.check_engine(here=none, engine=live)
    check("live copy" in line and live in line and "no pinned copy" in line, "check_engine: no pinned copy -> one line that names the trainkit in use")
    check("pinned copy" in T.check_engine(here=code, engine=os.path.join(code, "trainkit")), "check_engine: the pinned copy imported -> the line says so")
    # end to end, train.py as a script with a config that does not exist (the check comes before the config is read)
    live_ar = os.path.abspath(os.path.join(BF, "..", "autoresearch"))
    shutil.copy(os.path.join(BF, "training", "train.py"), os.path.join(code, "train.py"))
    refuse = ("import runpy, sys\nlive, siblings, script = sys.argv[1:4]\nsys.path[:0] = [live, siblings]\nsys.argv = [script, 'no_such_config.json']\n"
              "runpy.run_path(script, run_name='__main__')\n")
    p = subprocess.run([sys.executable, "-c", refuse, live_ar, os.path.join(BF, "training"), os.path.join(code, "train.py")], capture_output=True, text=True, cwd=BF)
    check(p.returncode != 0 and "a pinned trainkit lies next to it" in p.stderr and "no_such_config" not in p.stderr, "as a script: the live trainkit imported over a pinned copy -> refused before the config is read")
    p = subprocess.run([sys.executable, os.path.join("training", "train.py"), "no_such_config.json"], capture_output=True, text=True, cwd=BF)
    check(p.stdout.startswith("trainkit: the live copy ") and "no pinned copy" in p.stdout.splitlines()[0], "as a script, no pinned copy: the first line of the log names the live trainkit")
    shutil.rmtree(os.path.join(code, "trainkit"))
    shutil.copytree(os.path.join(live_ar, "trainkit"), os.path.join(code, "trainkit"), ignore=shutil.ignore_patterns("__pycache__"))
    p = subprocess.run([sys.executable, os.path.join(code, "train.py"), "no_such_config.json"], capture_output=True, text=True, cwd=BF,
                       env={**os.environ, "PYTHONPATH": os.path.join(BF, "training")})
    check(p.stdout.startswith("trainkit: the pinned copy " + os.path.join(code, "trainkit")), "as a script from a snapshot with the copy next to it: the pinned copy runs and the log says so")

print("TRAIN GUARDS: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)

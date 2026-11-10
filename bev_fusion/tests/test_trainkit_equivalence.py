"""The trainer after the move into autoresearch/trainkit (D48) gives the numbers of the trainer before it: the guard of the frozen trainer (FROZEN.md, changelog 2026-11-03).

The OLD trainer is `training/train.py` of commit OLD_COMMIT (taken from git at the start of the test), the NEW one is the live `training/train.py` with the engine in
`autoresearch/trainkit`. Both run the REAL `train()` on the CPU on a toy network (a frozen part, a batch norm, a Kendall-like loss parameter; the toy loss depends on step_frac
and draws once from torch's and numpy's global RNG per forward) with the same seed and the same batches, and must agree on everything a run leaves behind: the printed lines, every
row of log.jsonl (all but the wall-clock fields), the step_frac every forward saw, the weights, EMA, optimiser state, RNG state and scaler of every checkpoint (bitwise), the files in
the run folder, config.json, incident dumps, STOPPED, and the exception text of a refused or aborted start. Declared differences (ADDED, ADDED_VAL): the config fields added after the old
commit and the keys n / n_skipped of a validation row. The loader of the toy is a plain re-iterable (no `iter_from`), so the data position of a resume (trainkit.data) is not part of
this comparison: it is pinned by `python -m trainkit.selftest` and tests/test_train_guards.py.

  plain / frozen / bad / badfp16 / abort   probe, lr_groups, split clip, evaluation, pruning; train_only + fp16; non-finite steps, incidents; abort after max_bad_steps in a row
  crash, evalcrash                         a crash in the middle (log rows above the checkpoint are superseded, both RNG streams restored), a crash inside an evaluation
  stop, stopfp16, stopkeep                 a stop request (CONTROL file): checkpoint, STOPPED, exit 75, the older checkpoints pruned; the resume moves the old files aside
  savecrash                                a disk error during the checkpoint write: only a .tmp is left (no partial checkpoint), the resume removes it
  warmresume, warmslim                     init_from AND a checkpoint of the run: no warm start on a resume; warm start, slim finish, a refused resume and the override
  config                                   TrainConfig has the fields and defaults of the old one
  mutants                                  engine changes that alter a number or a file (listed in mutants()) must make a scenario differ; a mutant that survives fails the test
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_trainkit_equivalence.py   (CPU, about 15 s)
"""
import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
OLD_COMMIT = "df9d4370723116462e879a335da9d3ec83841229"            # the last commit of training/train.py before the engine moved out (hygiene flag, lr_groups)
WALL = ("t", "s_per_step", "t_data", "t_step")                   # wall-clock fields of a log row
ADDED = ("student_anchor", "student_anchor_floor", "student_anchor_until")      # TrainConfig fields added after OLD_COMMIT (t22 stage B, default off): the old trainer has none, so they are left out of every config compared
ADDED_VAL = ("n", "n_skipped")                                   # the keys a validation row has since 2026-11-10 (D73, review finding 2: the batches in the mean and the ones left out as non-finite); the old trainer has none, every value that it has stays compared
if os.path.isdir("/dev/shm"):
    tempfile.tempdir = "/dev/shm"                               # the run folders of the toy are small and written with fsync: keep them off the disk

import train as NEW  # noqa: E402 - the live trainer

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def load_old(tmp):
    """training/train.py of OLD_COMMIT as a module of its own (its siblings are imported from training/ as usual)"""
    src = subprocess.run(["git", "-C", ROOT, "show", f"{OLD_COMMIT}:bev_fusion/training/train.py"], capture_output=True, check=True).stdout
    path = os.path.join(tmp, "train_before_trainkit.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("train_before_trainkit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ the toy
class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.default_generator.manual_seed(0)                  # (torch.manual_seed would also queue a CUDA seeding with a stack trace: 4 ms a call)
        self.camera = nn.Linear(6, 6)                           # the frozen pre-trained part: not in the checkpoint
        for p in self.camera.parameters():
            p.requires_grad_(False)
        self.net = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 3))
        self.grid = None
        self.cfg = SimpleNamespace(freeze_camera=True)

    def forward(self, batch, dts=None):
        return self.net(batch["x"])


class ToyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))


FRACS = []                                                      # the step_frac of every forward call of the toy, compared old against new


def toy_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    FRACS.append(step_frac)
    h = model.net[2](model.net[1](model.net[0](batch["x"])))
    noise = 0.01 * (float(torch.rand(())) + float(np.random.rand()))     # one draw of each global RNG per forward: a resume that does not restore a stream changes every later row
    tot = ((model.net[3](h) - batch["y"]) ** 2).mean() + 0.01 * loss.s.pow(2).sum() + 0.1 * step_frac * model.net[3].bias.pow(2).sum() + noise     # step_frac has a gradient
    mode = batch.get("mode", "ok")
    if mode == "nan_loss":
        tot = tot * float("nan")
    elif mode == "nan_grad":                                     # a finite loss with a NaN gradient (sqrt at 0 times 0)
        tot = tot + (model.net[3].weight.abs().sum() * 0).sqrt()
    return {"total": tot, "group.a": tot.detach()}, {"state": {"m": h.detach()}}, None


def toy_batches(bad=None, n=40):
    g = torch.Generator().manual_seed(7)
    bad = bad or {}
    return [{"x": torch.randn(16, 6, generator=g), "y": torch.randn(16, 3, generator=g), "tokens": [f"t{i}"], "mode": bad.get(i, "ok")} for i in range(n)]


class Loader:
    """the training batches as a re-iterable; the n-th fetch of the call can raise (a crash) or write a CONTROL file (a stop request)"""

    def __init__(self, batches, run_dir, crash_at=None, stop_at=None):
        self.batches, self.run_dir, self.crash_at, self.stop_at, self.n = batches, run_dir, crash_at, stop_at, 0

    def __iter__(self):
        for b in self.batches:
            self.n += 1
            if self.n == self.crash_at:
                raise RuntimeError("simulated crash")
            if self.n == self.stop_at:
                with open(os.path.join(self.run_dir, "CONTROL"), "w") as f:
                    json.dump({"cmd": "stop", "reason": "equivalence test"}, f)
            yield b


def dying_save(n):
    """torch.save whose n-th call leaves half a file and raises (a full disk during the checkpoint write)"""
    real, count = torch.save, [0]

    def save(obj, f, *a, **k):
        count[0] += 1
        real(obj, f, *a, **k)
        if count[0] == n:
            with open(f, "r+b") as fh:
                fh.truncate(max(1, os.path.getsize(f) // 2))
            raise OSError("simulated: disk full during the checkpoint write")
    return save


# ema 0.3: the decay ramp (1 + step) / (10 + step) is capped by it from step 3 on, so both the cap and the configured value show in 12 steps
BASE = dict(steps=12, accum=2, warmup=3, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, max_bad_steps=3, seed=0, lr=1e-2, grad_clip=0.5, ema=0.3)


def call(T, root, name, cfg_kw=None, bad=None, crash_at=None, stop_at=None, eval_fn=None, env=None, save_dies=None):
    """one train() call of module T on the toy -> {out, exc, ret, fracs}; the build / forward / loader hooks of the module are replaced for the call"""
    keys = ("build", "forward_loss", "make_loader", "val_subset")
    saved = {k: getattr(T, k) for k in keys}
    T.build = lambda cfg: (ToyNet(), ToyLoss(), True)
    T.forward_loss = toy_forward
    T.make_loader = lambda ds, cfg, train, workers=None, generator=None: ds
    T.val_subset = lambda ds, n: ds
    real_save = torch.save
    if save_dies:
        torch.save = dying_save(save_dies)
    cfg = T.TrainConfig(**{**BASE, **(cfg_kw or {}), "name": name, "out_root": root})
    data = Loader(toy_batches(bad), os.path.join(root, name), crash_at, stop_at)
    env = env or {}
    os.environ.update(env)
    del FRACS[:]
    buf, exc, ret = io.StringIO(), None, None
    try:
        with contextlib.redirect_stdout(buf):
            model, loss, ema, last = T.train(cfg, eval_fn=eval_fn, datasets=(data, toy_batches()[:6]))
        ret = (model.state_dict(), loss.state_dict(), ema.shadow, {k: v for k, v in last.items() if k not in WALL})
    except BaseException as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e.code if isinstance(e, SystemExit) else e}"
    finally:
        torch.save = real_save
        for k, v in saved.items():
            setattr(T, k, v)
        for k in env:
            os.environ.pop(k, None)
    return dict(out=buf.getvalue().replace(root, "<root>"), exc=None if exc is None else exc.replace(root, "<root>"), ret=ret, fracs=list(FRACS))


def snapshot(root, name):
    """what the run folder holds afterwards"""
    d = os.path.join(root, name)
    files = sorted(f for f in os.listdir(d) if f != "__pycache__")

    def rows(fn):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            return None
        out = []
        for ln in open(p):
            r = json.loads(ln) if ln.strip().endswith("}") else {"cut": ln}
            if isinstance(r.get("val"), dict):
                r["val"] = {k: v for k, v in r["val"].items() if k not in ADDED_VAL}
            out.append(json.dumps({k: v for k, v in r.items() if k not in WALL}, sort_keys=True))
        return out
    here = lambda x: {k: v for k, v in json.loads(json.dumps(x, default=str).replace(root.replace("\\", "\\\\"), "<root>")).items() if k not in ADDED}      # noqa: E731 - the two roots differ: compare the rest of every path
    snap = dict(files=files, log=rows("log.jsonl"), superseded=rows("log_superseded.jsonl"),
                config=here(json.load(open(os.path.join(d, "config.json")))) if "config.json" in files else None)
    snap["stopped"] = {k: v for k, v in json.load(open(os.path.join(d, "STOPPED"))).items() if k != "t"} if "STOPPED" in files else None
    snap["incidents"] = {f: json.dumps({k: v for k, v in json.load(open(os.path.join(d, f))).items() if k != "time"}, sort_keys=True) for f in files if f.startswith("incident_")}
    snap["ckpts"] = {}
    for f in files:                                              # a partial file (a write that died in place) is no checkpoint: its size stands for it
        if f.endswith(".pt"):
            try:
                snap["ckpts"][f] = torch.load(os.path.join(d, f), map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001
                snap["ckpts"][f] = f"unreadable, {os.path.getsize(os.path.join(d, f))} bytes"
    for ck in snap["ckpts"].values():
        if isinstance(ck, dict):
            ck["cfg"] = here(ck["cfg"])
    return snap


def diff(a, b, path=""):
    """the first difference of two trees (bitwise for tensors and arrays), else None"""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(set(a) ^ set(b))[:4]}"
        return next((d for k in a if (d := diff(a[k], b[k], f"{path}/{k}"))), None)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} / {len(b)}"
        return next((d for i, (x, y) in enumerate(zip(a, b)) if (d := diff(x, y, f"{path}[{i}]"))), None)
    if torch.is_tensor(a) and torch.is_tensor(b):
        return None if (a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)) else f"{path}: tensors differ"
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return None if np.array_equal(a, b) else f"{path}: arrays differ"
    if isinstance(a, str) and isinstance(b, str) and a != b and "\n" in a + b:             # printed output: name the first line that differs
        i = next(k for k, (x, y) in enumerate(zip(a.split("\n") + [""], b.split("\n") + [""])) if x != y)
        return f"{path}: line {i}: {(a.split(chr(10)) + ['])[i][:70]!r} / {(b.split(chr(10)) + ['])[i][:70]!r}"
    return None if (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)) else f"{path}: {str(a)[:60]!r} / {str(b)[:60]!r}"


def run(T, root, calls):
    """the calls one after the other, each in the run folder it names ("<A>" as init_from is the first checkpoint of folder a)
    -> ([result per call], [what its run folder holds after each call]): the state after a crash or a stop is compared, not only the end"""
    res, snaps = [], []
    for c in calls:
        c = dict(c)
        name = c.pop("name")
        if (c.get("cfg_kw") or {}).get("init_from") == "<A>":
            c["cfg_kw"] = dict(c["cfg_kw"], init_from=os.path.join(root, "a", "ckpt_00000004.pt"))
        res.append(call(T, root, name, **c))
        snaps.append(snapshot(root, name))
    return res, snaps


# ------------------------------------------------------------------------------------------------ the scenarios
def S(name, **kw):
    return dict(kw, name=name)


PLAIN = dict(cfg_kw=dict(probe=True, eval_every=6, keep=2, grad_clip_split=("net.3",), lr_groups={"net.0": {"lr": 5e-3, "warmup": 4}}))
CRASH = dict(cfg_kw=dict(eval_every=4))
STOPKEEP = dict(cfg_kw=dict(save_every=2, keep=1))
WARM = dict(cfg_kw=dict(steps=8, init_from="<A>"))


def boom(model, loader):
    raise MemoryError("simulated ENOMEM in a validation worker")


# (key, what it shows, the calls). The order is the order of the report; KILL names the ones the mutants are run against (cheapest and most telling first)
SCENARIOS = [
    ("plain", "plain run: probe, lr_groups, split clip, evaluation, pruning", [S("plain", **PLAIN)]),
    ("frozen", "train_only (EMA skip of the frozen parts) and fp16 with a GradScaler", [S("frozen", cfg_kw=dict(train_only=("net.3",), amp="fp16", save_every=6))]),
    ("bad", "non-finite steps skipped (nan loss, nan gradient), probe incidents",
     [S("bad", cfg_kw=dict(probe=True, steps=10), bad={4: "nan_loss", 5: "nan_loss", 14: "nan_grad"})]),
    ("badfp16", "the same under fp16 (an overflow halves the scale, a nan loss counts)",
     [S("badfp16", cfg_kw=dict(probe=True, steps=10, amp="fp16"), bad={4: "nan_loss", 14: "nan_grad", 15: "nan_grad"})]),
    ("abort", "too many non-finite steps in a row abort the run", [S("abort", cfg_kw=dict(steps=10), bad={i: "nan_loss" for i in range(6, 14)})]),
    ("crash", "a crash after step 8, then the resume (rows above the checkpoint are superseded, both RNG streams restored)",
     [S("crash", crash_at=19, **CRASH), S("crash", **CRASH)]),
    ("evalcrash", "a crash inside the evaluation of step 4, the evaluation is made up on the resume",
     [S("evalcrash", cfg_kw=dict(eval_every=4, steps=8), eval_fn=boom), S("evalcrash", cfg_kw=dict(eval_every=4, steps=8))]),
    ("stop", "a stop request: checkpoint, STOPPED, exit 75; the resume moves the old files aside and finishes",
     [S("stop", cfg_kw=dict(save_every=0), stop_at=11), S("stop", cfg_kw=dict(save_every=0))]),
    ("stopfp16", "a stop request under fp16 keeps the scaler in the checkpoint",
     [S("stopfp16", cfg_kw=dict(save_every=0, amp="fp16"), stop_at=7), S("stopfp16", cfg_kw=dict(save_every=0, amp="fp16"))]),
    ("stopkeep", "a stop at step 7 with a checkpoint every 2 steps and keep 1: the stop checkpoint prunes the older one",
     [S("stopkeep", stop_at=13, **STOPKEEP), S("stopkeep", **STOPKEEP)]),
    ("savecrash", "a disk error during the checkpoint write of step 8: only a .tmp is left, no partial checkpoint; the resume (from step 4) removes it",
     [S("savecrash", cfg_kw=dict(save_every=4), save_dies=2), S("savecrash", cfg_kw=dict(save_every=0))]),
    ("warmresume", "init_from AND a checkpoint of the run (crash after step 4): the resume does not warm start again",
     [S("a", cfg_kw=dict(steps=4, save_every=2)), S("wr", crash_at=11, **WARM), S("wr", **WARM)]),
    ("warmslim", "warm start (init_from, init_reset), slim finish, the finished slim run again, a refused resume and the override",
     [S("a", cfg_kw=dict(steps=4, save_every=2)), S("b", cfg_kw=dict(steps=8, init_from="<A>", init_reset=("net.3.bias",), slim_at_end=True)),
      S("b", cfg_kw=dict(steps=8, slim_at_end=False)), S("b", cfg_kw=dict(steps=12)), S("b", cfg_kw=dict(steps=12), env={"BF_RESUME_ANY": "1"})]),
]
KILL = ["plain", "crash", "stop", "stopkeep", "savecrash", "warmresume", "bad", "abort"]


def play(T, calls):
    with tempfile.TemporaryDirectory() as root:
        return run(T, root, calls)


def scenarios(OLD):
    for key, label, calls in SCENARIOS:
        d = diff(play(OLD, calls), play(NEW, calls))
        check(d is None, label + ("" if d is None else f"  [{key}: {d}]"))


def config_fields(OLD):
    a, b = dataclasses.asdict(OLD.TrainConfig()), {k: v for k, v in dataclasses.asdict(NEW.TrainConfig()).items() if k not in ADDED}
    d = diff(a, b) or diff(sorted(OLD.RESUME_FREE), sorted(NEW.RESUME_FREE))
    check(d is None, "TrainConfig: the same fields and defaults and the same RESUME_FREE as before" + ("" if d is None else f"  [{d}]"))


# ------------------------------------------------------------------------------------------------ the mutants
def mutated(module, old, new, count=1):
    """a copy of the module whose source has `old` replaced by `new`; `old` must occur `count` times, so a refactor that moves the line fails the test instead of making the mutant harmless"""
    src = open(module.__file__, encoding="utf-8").read()
    assert src.count(old) == count, f"{module.__name__}: the text to mutate occurs {src.count(old)} times, not {count}: {old[:70]!r}"
    copy = types.ModuleType(module.__name__ + "_mutant")
    copy.__package__, copy.__file__ = module.__package__, module.__file__
    exec(compile(src.replace(old, new), module.__file__, "exec"), copy.__dict__)
    return copy


def mutants():
    """engine changes that alter a number or a file: (label, [(object, attribute, replacement)]). The patched names are the ones the loop looks up at call time."""
    import trainkit.checkpoint as ck
    import trainkit.control as ctl
    import trainkit.ema as em
    import trainkit.loop as lp
    import trainkit.optim as op
    import trainkit.runlog as rl
    import trainkit.schedule as sc
    ema_update, lr_factor = em.EMA.update, sc.lr_factor

    def fit(old, new, count=1):
        return [(NEW, "fit", mutated(lp, old, new, count).fit)]          # train() calls the module's `fit`

    def ema_off_by_one(self, model, step):
        ema_update(self, model, step + 1)

    def no_clip(params, grad_clip, split=()):
        return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))

    stop_block = "            save(step)\n            checkpoint.prune_checkpoints(out_dir, cfg.keep)\n            control.mark_stopped"
    tmp_prune = 'if f.startswith("ckpt_") and f.endswith(".pt.tmp"):\n            os.remove(os.path.join(out_dir, f))'
    return [
        ("EMA decay one step off", [(em.EMA, "update", ema_off_by_one)]),
        ("EMA without the early ramp cap min(decay, ...)", [(em.EMA, "update", mutated(em, "d = min(self.decay, (1 + step) / (10 + step))", "d = self.decay").EMA.update)]),
        ("EMA built with 0.95, not the configured ema", fit("EMA(model, cfg.ema, skip=frozen)", "EMA(model, 0.95, skip=frozen)", 2)),
        ("lr schedule shifted by one step", [(sc, "lr_factor", lambda step, warmup, total, min_frac: lr_factor(step + 1, warmup, total, min_frac))]),
        ("step_frac is 0 for the step function", fit("StepInfo(step, step / max(1, cfg.steps), m, cfg.accum, scaler)", "StepInfo(step, 0.0, m, cfg.accum, scaler)")),
        ("step_frac is (step + 1) / steps", fit("StepInfo(step, step / max(1, cfg.steps), m, cfg.accum, scaler)", "StepInfo(step, (step + 1) / cfg.steps, m, cfg.accum, scaler)")),
        ("gradient clip skipped", [(op, "clip_gradients", no_clip)]),
        ("a bad step does not count", [(op, "bad_steps_after", lambda status, bad, scaler=None: 0)]),
        ("old checkpoints are not pruned", [(ck, "prune_checkpoints", lambda out_dir, keep: None)]),
        ("no prune after the stop checkpoint", fit(stop_block, "            save(step)\n            control.mark_stopped")),
        ("prune leaves the .tmp of a failed write", [(ck, "prune_checkpoints", mutated(ck, tmp_prune, tmp_prune.replace("os.remove(os.path.join(out_dir, f))", "pass")).prune_checkpoints)]),
        ("checkpoint written in place, not through a .tmp", [(ck, "save_checkpoint", mutated(ck, 'tmp = path + ".tmp"', "tmp = path").save_checkpoint)]),
        ("log rows above the resume step stay", [(rl, "supersede_log", lambda out_dir, step: None)]),
        ("a stopped run exits with 0", [(ctl, "EXIT_STOPPED", 0)]),
        ("the checkpoint forgets the RNG state", [(torch, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8))]),
        ("torch RNG state saved but not restored", [(torch, "set_rng_state", lambda state: None)]),
        ("numpy RNG state saved but not restored", [(np.random, "set_state", lambda state: None)]),
        ("a warm start also on a resume", fit("if resumed is None and cfg.init_from:", "if cfg.init_from:")),
    ]


def kill_mutants(OLD):
    """every mutant must make at least one of the KILL scenarios differ from the old trainer"""
    table = {key: calls for key, _, calls in SCENARIOS}
    refs = {key: play(OLD, table[key]) for key in KILL}
    for label, patches in mutants():
        saved = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
        for obj, attr, new in patches:
            setattr(obj, attr, new)
        killed = None
        try:
            for key in KILL:
                d = diff(refs[key], play(NEW, table[key]))
                if d:
                    killed = f"{key}: {d}"
                    break
        finally:
            for obj, attr, orig in saved:
                setattr(obj, attr, orig)
        check(killed is not None, f"mutant killed: {label}" + (f"  [{killed[:100]}]" if killed else "  -- SURVIVED"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        OLD_T = load_old(tmp)
        config_fields(OLD_T)
        scenarios(OLD_T)
        kill_mutants(OLD_T)
    print("TRAINKIT EQUIVALENCE GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)

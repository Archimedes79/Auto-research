"""The trainer after the move into autoresearch/trainkit (D48) gives the numbers of the trainer before it: the guard of the frozen trainer (FROZEN.md, changelog 2026-11-03).

The OLD trainer is `training/train.py` of commit OLD_COMMIT (taken from git at the start of the test), the NEW one is the live `training/train.py` with the engine in
`autoresearch/trainkit`. Both run the REAL `train()` on the CPU on a toy network (a frozen part, a batch norm, a Kendall-like loss parameter) with the same seed and the same
batches, and must agree on everything a run leaves behind: the printed lines, every row of log.jsonl (all but the wall-clock fields), the weights, EMA, optimiser state, RNG state and
scaler of every checkpoint (bitwise), the files in the run folder, config.json, incident dumps, STOPPED, and the exception text of a refused or aborted start.

  1 a plain run with probe, lr_groups, split clip, evaluation every 6 steps, pruning (keep 2)    2 a run with train_only (frozen parts: EMA skip) and fp16 (GradScaler)
  3 non-finite steps (nan loss, nan gradient) under fp16 and without, probe incidents, then an abort after max_bad_steps in a row
  4 a crash in the middle (log rows above the checkpoint move to log_superseded.jsonl) and the resume;   a crash inside an evaluation, made up on the resume
  5 a stop request (CONTROL file): checkpoint, STOPPED, exit 75, no .done; the resume moves the old files aside and finishes
  6 a warm start (init_from, init_reset), a slim finish, a finished slim run started again, a resume with another config (refused; the override continues)
  7 the config: TrainConfig has the fields and defaults of the old one
  8 the mutants: engine changes that alter a number (EMA decay, schedule, clip, bad-step rule, pruning, log supersede, stop exit code, checkpoint contents) must make the comparison fail
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_trainkit_equivalence.py   (CPU, about 1 min)
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
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
OLD_COMMIT = "df9d4370723116462e879a335da9d3ec83841229"            # the last commit of training/train.py before the engine moved out (hygiene flag, lr_groups)
WALL = ("t", "s_per_step", "t_data", "t_step")                   # wall-clock fields of a log row

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
        torch.manual_seed(0)
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


def toy_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    h = model.net[2](model.net[1](model.net[0](batch["x"])))
    tot = ((model.net[3](h) - batch["y"]) ** 2).mean() + 0.01 * loss.s.pow(2).sum()
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


BASE = dict(steps=12, accum=2, warmup=3, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, max_bad_steps=3, seed=0, lr=1e-2, grad_clip=0.5, ema=0.9)


def call(T, root, name, cfg_kw=None, bad=None, crash_at=None, stop_at=None, eval_fn=None, env=None):
    """one train() call of module T on the toy -> {out, exc, ret}; the build / forward / loader hooks of the module are replaced for the call"""
    keys = ("build", "forward_loss", "make_loader", "val_subset")
    saved = {k: getattr(T, k) for k in keys}
    T.build = lambda cfg: (ToyNet(), ToyLoss(), True)
    T.forward_loss = toy_forward
    T.make_loader = lambda ds, cfg, train, workers=None, generator=None: ds
    T.val_subset = lambda ds, n: ds
    cfg = T.TrainConfig(**{**BASE, **(cfg_kw or {}), "name": name, "out_root": root})
    data = Loader(toy_batches(bad), os.path.join(root, name), crash_at, stop_at)
    env = env or {}
    os.environ.update(env)
    buf, exc, ret = io.StringIO(), None, None
    try:
        with contextlib.redirect_stdout(buf):
            model, loss, ema, last = T.train(cfg, eval_fn=eval_fn, datasets=(data, toy_batches()[:6]))
        ret = (model.state_dict(), loss.state_dict(), ema.shadow, {k: v for k, v in last.items() if k not in WALL})
    except BaseException as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e.code if isinstance(e, SystemExit) else e}"
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
        for k in env:
            os.environ.pop(k, None)
    return dict(out=buf.getvalue().replace(root, "<root>"), exc=None if exc is None else exc.replace(root, "<root>"), ret=ret)


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
            out.append(json.dumps({k: v for k, v in r.items() if k not in WALL}, sort_keys=True))
        return out
    here = lambda x: json.loads(json.dumps(x, default=str).replace(root.replace("\\", "\\\\"), "<root>"))      # noqa: E731 - the two roots differ: compare the rest of every path
    snap = dict(files=files, log=rows("log.jsonl"), superseded=rows("log_superseded.jsonl"),
                config=here(json.load(open(os.path.join(d, "config.json")))) if "config.json" in files else None)
    snap["stopped"] = {k: v for k, v in json.load(open(os.path.join(d, "STOPPED"))).items() if k != "t"} if "STOPPED" in files else None
    snap["incidents"] = {f: json.dumps({k: v for k, v in json.load(open(os.path.join(d, f))).items() if k != "time"}, sort_keys=True) for f in files if f.startswith("incident_")}
    snap["ckpts"] = {f: torch.load(os.path.join(d, f), map_location="cpu", weights_only=False) for f in files if f.endswith(".pt")}
    for ck in snap["ckpts"].values():
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
    return None if (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)) else f"{path}: {str(a)[:60]!r} / {str(b)[:60]!r}"


def play(T, root, name, calls):
    """the calls (dicts of call() arguments) one after the other in the same run folder -> ([result per call], folder snapshot)"""
    res = [call(T, root, name, **kw) for kw in calls]
    return res, snapshot(root, name)


def same(old, new, label):
    """compare two play() results; True when identical, else print the first difference"""
    d = diff(old, new)
    check(d is None, label + ("" if d is None else f"  [{d}]"))
    return d is None


def case(OLD, label, name, calls):
    with tempfile.TemporaryDirectory() as ro, tempfile.TemporaryDirectory() as rn:
        o = play(OLD, ro, name, [dict(c) for c in calls])
        n = play(NEW, rn, name, [dict(c) for c in calls])
    return same(o, n, label)


# ------------------------------------------------------------------------------------------------ the cases
PLAIN = dict(cfg_kw=dict(probe=True, eval_every=6, keep=2, grad_clip_split=("net.3",), lr_groups={"net.0": {"lr": 5e-3, "warmup": 4}}))
FROZEN_FP16 = dict(cfg_kw=dict(train_only=("net.3",), amp="fp16", save_every=6))
NONFINITE = dict(cfg_kw=dict(probe=True, steps=10, max_bad_steps=3), bad={4: "nan_loss", 5: "nan_loss", 14: "nan_grad"})
NONFINITE_FP16 = dict(cfg_kw=dict(probe=True, steps=10, max_bad_steps=3, amp="fp16"), bad={4: "nan_loss", 14: "nan_grad", 15: "nan_grad"})
ABORT = dict(cfg_kw=dict(steps=10, max_bad_steps=3), bad={i: "nan_loss" for i in range(6, 14)})


def cases(OLD):
    ok = True
    ok &= case(OLD, "plain run: probe, lr_groups, split clip, evaluation, pruning", "plain", [PLAIN])
    ok &= case(OLD, "train_only (EMA skip of the frozen parts) and fp16 with a GradScaler", "frozen", [FROZEN_FP16])
    ok &= case(OLD, "non-finite steps skipped (nan loss, nan gradient), probe incidents", "bad", [NONFINITE])
    ok &= case(OLD, "the same under fp16 (an overflow halves the scale, a nan loss counts)", "badfp16", [NONFINITE_FP16])
    ok &= case(OLD, "too many non-finite steps in a row abort the run", "abort", [ABORT])
    ok &= case(OLD, "a crash after step 8, then the resume (rows above the checkpoint are superseded)", "crash",
               [dict(cfg_kw=dict(eval_every=4), crash_at=19), dict(cfg_kw=dict(eval_every=4))])

    def boom(model, loader):
        raise MemoryError("simulated ENOMEM in a validation worker")
    ok &= case(OLD, "a crash inside the evaluation of step 4, the evaluation is made up on the resume", "evalcrash",
               [dict(cfg_kw=dict(eval_every=4, steps=8), eval_fn=boom), dict(cfg_kw=dict(eval_every=4, steps=8))])
    ok &= case(OLD, "a stop request: checkpoint, STOPPED, exit 75; the resume moves the old files aside and finishes", "stop",
               [dict(cfg_kw=dict(save_every=0), stop_at=11), dict(cfg_kw=dict(save_every=0))])
    ok &= case(OLD, "a stop request under fp16 keeps the scaler in the checkpoint", "stopfp16",
               [dict(cfg_kw=dict(save_every=0, amp="fp16"), stop_at=7), dict(cfg_kw=dict(save_every=0, amp="fp16"))])
    return ok


def warm_and_slim(OLD):
    """a warm start from the first run, a slim finish, a finished slim run started again, a resume with another config"""
    def calls():
        first = dict(cfg_kw=dict(steps=4, save_every=2))
        return [dict(first, name="a"), dict(cfg_kw=dict(steps=8, init_from="<A>", init_reset=("net.3.bias",), slim_at_end=True), name="b"),
                dict(cfg_kw=dict(steps=8, slim_at_end=False), name="b"), dict(cfg_kw=dict(steps=12), name="b"),
                dict(cfg_kw=dict(steps=12), name="b", env={"BF_RESUME_ANY": "1"})]
    with tempfile.TemporaryDirectory() as ro, tempfile.TemporaryDirectory() as rn:
        out = []
        for T, root in ((OLD, ro), (NEW, rn)):
            res = []
            for c in calls():
                c = dict(c)
                name = c.pop("name")
                if c["cfg_kw"].get("init_from") == "<A>":
                    c["cfg_kw"] = dict(c["cfg_kw"], init_from=os.path.join(root, "a", "ckpt_00000004.pt"))
                res.append(call(T, root, name, **c))
            out.append((res, snapshot(root, "a"), snapshot(root, "b")))
    o, n = out
    return same(o, n, "warm start (init_from, init_reset), slim finish, the finished slim run again, a refused resume and the override")


def config_fields(OLD):
    a, b = dataclasses.asdict(OLD.TrainConfig()), dataclasses.asdict(NEW.TrainConfig())
    return same(a, b, "TrainConfig: the same fields and defaults as before") and same(sorted(OLD.RESUME_FREE), sorted(NEW.RESUME_FREE), "RESUME_FREE: the same names")


# ------------------------------------------------------------------------------------------------ the mutants
def mutants():
    """engine changes that alter a number: (label, [(object, attribute, replacement)]). The patched names are the ones the loop looks up at call time."""
    import trainkit.checkpoint as ck
    import trainkit.control as ctl
    import trainkit.ema as em
    import trainkit.optim as op
    import trainkit.runlog as rl
    import trainkit.schedule as sc
    ema_update, lr_factor = em.EMA.update, sc.lr_factor

    def ema_off_by_one(self, model, step):
        ema_update(self, model, step + 1)

    def no_clip(params, grad_clip, split=()):
        return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))

    def prune_nothing(out_dir, keep):
        return None

    return [("EMA decay one step off", [(em.EMA, "update", ema_off_by_one)]),
            ("lr schedule shifted by one step", [(sc, "lr_factor", lambda step, warmup, total, min_frac: lr_factor(step + 1, warmup, total, min_frac))]),
            ("gradient clip skipped", [(op, "clip_gradients", no_clip)]),
            ("a bad step does not count", [(op, "bad_steps_after", lambda status, bad, scaler=None: 0)]),
            ("old checkpoints are not pruned", [(ck, "prune_checkpoints", prune_nothing)]),
            ("log rows above the resume step stay", [(rl, "supersede_log", lambda out_dir, step: None)]),
            ("a stopped run exits with 0", [(ctl, "EXIT_STOPPED", 0)]),
            ("the checkpoint forgets the RNG state", [(torch, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8))])]


def kill_mutants(OLD):
    """every mutant must make at least one of the scenarios differ from the old trainer"""
    scenarios = [("plain", [PLAIN]), ("bad", [NONFINITE]), ("abort", [ABORT]), ("crash", [dict(cfg_kw=dict(eval_every=4), crash_at=19), dict(cfg_kw=dict(eval_every=4))]),
                 ("stop", [dict(cfg_kw=dict(save_every=0), stop_at=11), dict(cfg_kw=dict(save_every=0))])]
    refs = {}
    for name, calls in scenarios:
        with tempfile.TemporaryDirectory() as ro:
            refs[name] = play(OLD, ro, name, [dict(c) for c in calls])
    for label, patches in mutants():
        saved = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
        for obj, attr, new in patches:
            setattr(obj, attr, new)
        killed = None
        try:
            for name, calls in scenarios:
                with tempfile.TemporaryDirectory() as rn:
                    got = play(NEW, rn, name, [dict(c) for c in calls])
                d = diff(refs[name], got)
                if d:
                    killed = f"{name}: {d}"
                    break
        finally:
            for obj, attr, orig in saved:
                setattr(obj, attr, orig)
        check(killed is not None, f"mutant killed: {label}" + (f"  [{killed[:90]}]" if killed else "  -- SURVIVED"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        OLD_T = load_old(tmp)
        config_fields(OLD_T)
        cases(OLD_T)
        warm_and_slim(OLD_T)
        if importlib.util.find_spec("trainkit.loop") is not None:       # the loop is the engine's (stage B); before that the loop of train.py calls aliases the mutants cannot reach
            kill_mutants(OLD_T)
    print("TRAINKIT EQUIVALENCE GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)

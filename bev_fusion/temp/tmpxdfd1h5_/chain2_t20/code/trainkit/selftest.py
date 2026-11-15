"""Self-test of the engine on a toy (CPU, a few seconds): known answers for the schedule, the EMA, the clip and step rules, the resume rules and the stop protocol, and the
example of README.md end to end (gradient reaches every parameter, accumulation, checkpoints, resume, stop, hooks).
Run: python -m trainkit.selftest    (from autoresearch/)"""
import contextlib
import dataclasses
import io
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trainkit import checkpoint, control, optim, schedule  # noqa: E402
from trainkit.config import Config  # noqa: E402
from trainkit.data import EpochLoader  # noqa: E402
from trainkit.ema import EMA  # noqa: E402
from trainkit.loop import Hooks, StepInfo, fit  # noqa: E402
from trainkit.util import prepare_run_dir, seed_all  # noqa: E402


class Net(nn.Module):                                    # the MODEL
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(4, 1)


class Loss(nn.Module):                                   # the LOSS: a learnable weight of the squared error (a Kendall-like term)
    def __init__(self):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(()))

    def forward(self, pred, y):
        return ((pred - y) ** 2).mean() * torch.exp(-self.log_w) + self.log_w


def toy_data(n=10):                                      # the DATA: any re-iterable of batches
    g = torch.Generator().manual_seed(1)
    return [{"x": torch.randn(8, 4, generator=g), "y": torch.randn(8, 1, generator=g)} for _ in range(n)]


def stop_after(n, reason="selftest"):                  # a stop_check that asks for a stop from its (n + 1)-th call on: the run stops at the boundary of step n
    calls = [0]

    def stop_check(run_dir):
        calls[0] += 1
        return reason if calls[0] > n else None
    return stop_check


def run(tmp, name="toy", batches=None, stop_check=None, hooks=None, step_fn=None, fracs=None, **kw):
    cfg = Config(**{**dict(name=name, out_root=tmp, steps=12, accum=2, warmup=2, lr=0.05, device="cpu", log_every=4, save_every=6, eval_every=0), **kw})
    model, loss = Net(), Loss()

    def default_step(batch, info: StepInfo):             # one micro-batch: forward, backward (info.backward divides by accum), the scalar terms, finite?
        if fracs is not None:
            fracs.append(info.frac)
        l = loss(model.fc(batch["x"]), batch["y"])
        info.backward(l)
        return {"total": float(l.detach())}, bool(torch.isfinite(l)), None
    out = fit(cfg, model, loss, toy_data() if batches is None else batches, step_fn or default_step, prepare_run_dir(cfg), "cpu", stop_check=stop_check, hooks=hooks)
    return cfg, model, loss, out


def resume_toy(tmp, name, plain=False, stop=None, n=40, batch_size=4):
    """24 steps (accum 2) on shuffled data, whose step draws a random mask from the global stream; stopped after `stop` steps -> None, else (model, loss, EMA, the logged totals).
    The data order of `plain` is the global stream's (a plain DataLoader), the other one's is a function of (seed, epoch) (trainkit.data)"""
    cfg = Config(name=name, out_root=tmp, steps=24, accum=2, warmup=2, lr=0.05, device="cpu", log_every=4, save_every=6, eval_every=0)
    seed_all(cfg.seed)                                   # as train.py: the seed first, then the model, then the loaders
    model, loss = Net(), Loss()
    g = torch.Generator().manual_seed(1)
    data = torch.utils.data.TensorDataset(torch.randn(n, 4, generator=g), torch.randn(n, 1, generator=g))
    batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True) if plain else EpochLoader(data, batch_size, cfg.seed, drop_last=True)

    def step_fn(batch, info):
        x, y = batch
        l = loss(model.fc(x * (torch.rand(x.shape[0], 1) > 0.2).float()), y)       # like the BEV modality dropout: a random op on the training path
        info.backward(l)
        return {"total": float(l.detach())}, bool(torch.isfinite(l)), None
    try:
        _, _, ema, _ = fit(cfg, model, loss, batches, step_fn, prepare_run_dir(cfg), "cpu", stop_check=stop_after(stop) if stop else None)
    except SystemExit as e:
        assert e.code == control.EXIT_STOPPED
        return None
    rows = [json.loads(ln) for ln in open(os.path.join(tmp, name, "log.jsonl"))]
    return model, loss, ema, [r["total"] for r in rows]


def same_run(a, b):
    """two toy runs end in the same weights, EMA and logged totals, bit for bit"""
    return all(torch.equal(p, q) for p, q in zip(list(a[0].parameters()) + list(a[1].parameters()), list(b[0].parameters()) + list(b[1].parameters()))) \
        and all(torch.equal(a[2].shadow[k], b[2].shadow[k]) for k in a[2].shadow) and a[3] == b[3]


def check(cond, msg):
    assert cond, msg
    print("  ok  " + msg)


def main():
    # the schedule: from lr / 3 to lr in the warm-up, then cosine down to min_frac
    check(abs(schedule.lr_factor(0, 10, 100, 0.01) - (1 / 3 + 2 / 30)) < 1e-12 and abs(schedule.lr_factor(9, 10, 100, 0.01) - 1) < 1e-12, "lr_factor: warm-up from lr / 3 to lr")
    check(abs(schedule.lr_factor(55, 10, 100, 0.01) - (0.01 + 0.99 * 0.5)) < 1e-12 and abs(schedule.lr_factor(100, 10, 100, 0.01) - 0.01) < 1e-12, "lr_factor: halfway the cosine is at the middle, the end at min_frac")
    # the EMA: d = min(decay, (1 + step) / (10 + step)); the shadow of a frozen key stays the value itself
    net = nn.Linear(1, 1)
    ema = EMA(net, 0.9, skip={"bias"})
    w0, b0 = net.weight.item(), net.bias.item()
    with torch.no_grad():
        net.weight.fill_(3.0)
        net.bias.fill_(5.0)
    ema.update(net, 0)
    check(abs(ema.shadow["weight"].item() - (0.1 * w0 + 0.9 * 3.0)) < 1e-6 and ema.shadow["bias"].item() == b0, "EMA: decay ramp (1 + step) / (10 + step), skipped key untouched")
    with ema.applied(net):
        check(abs(net.weight.item() - ema.shadow["weight"].item()) < 1e-7 and not net.training, "EMA.applied: the average in eval mode ...")
    check(net.weight.item() == 3.0 and net.training, "... and the training weights back afterwards")
    e3 = EMA(net, 0.3)                                   # the shadow starts at the weight, 3.0
    with torch.no_grad():
        net.weight.fill_(7.0)
    e3.update(net, 5)                                    # the ramp says 6 / 15 = 0.4, the configured decay caps it at 0.3
    check(abs(e3.shadow["weight"].item() - (0.3 * 3.0 + 0.7 * 7.0)) < 1e-6, "EMA: the ramp is capped by the configured decay")
    # clip and step rules
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    opt = torch.optim.SGD([p], lr=1.0)
    gn, status = optim.optimizer_step(opt, [p], 1.0)
    check(abs(gn.item() - 5.0) < 1e-6 and status == "ok" and abs(p.detach().norm().item() - 1.0) < 1e-6, "optimizer_step: the norm before the clip is returned, the step used the clipped gradient")
    p.grad = torch.tensor([float("nan"), 1.0])
    check(optim.optimizer_step(opt, [p], 1.0)[1] == "nonfinite" and p.grad is None and bool(torch.isfinite(p).all()), "optimizer_step: a non-finite gradient is dropped, the weights stay")
    check([optim.bad_steps_after(s, 2) for s in ("ok", "nonfinite")] == [0, 3], "bad_steps_after: ok resets, a bad step counts")
    try:
        optim.param_groups(Net(), Loss(), 1e-3, 0.0, {"nope": {"lr": 1e-3}})
        raise AssertionError("an lr_groups prefix that names nothing must raise")
    except ValueError:
        print("  ok  lr_groups: a prefix that names no parameter is an error")
    # the stop protocol
    with tempfile.TemporaryDirectory() as tmp:
        check(control.stop_requested(tmp) is None, "control: no CONTROL file, no stop")
        control.request_stop(tmp, "why")
        check(control.stop_requested(tmp) == "why" and control.clear_stale(tmp) == ["CONTROL"] and control.stop_requested(tmp) is None, "control: a request is seen, the next start moves it aside")
        # the example: gradient reaches every parameter, the cadence of logs and checkpoints
        fracs = []
        cfg, model, loss, (_, _, _, last) = run(tmp, fracs=fracs)
        d = os.path.join(tmp, "toy")
        rows = [json.loads(ln) for ln in open(os.path.join(d, "log.jsonl"))]
        check([r["step"] for r in rows] == [1, 4, 8, 12] and last["step"] == 12 and all(math.isfinite(r["total"]) for r in rows), "fit: rows at step 1 and every log_every, the last row is returned")
        ref_model, ref_loss = Net(), Loss()
        check(all(not torch.equal(a, b) for a, b in list(zip(model.parameters(), ref_model.parameters())) + list(zip(loss.parameters(), ref_loss.parameters()))), "fit: every parameter of the model AND of the loss moved (the gradient reaches all)")
        check(sorted(f for f in os.listdir(d) if f.endswith(".pt")) == ["ckpt_00000006.pt", "ckpt_00000012.pt"] and os.path.exists(os.path.join(d, "config.json")), "fit: a checkpoint every save_every steps, the config copy")
        check(fracs == [s / 12 for s in range(12) for _ in range(2)], "fit: step_fn is told step / steps, the position in the schedule, for each of the 2 micro-batches")
        open(os.path.join(d, "ckpt_00000009.pt.tmp"), "wb").write(b"half a checkpoint")     # what a write that died leaves behind
        checkpoint.prune_checkpoints(d, 5)
        check(sorted(f for f in os.listdir(d) if f.startswith("ckpt_")) == ["ckpt_00000006.pt", "ckpt_00000012.pt"], "prune_checkpoints: the leftover .tmp of a failed write goes, the checkpoints stay")
        o = torch.optim.AdamW(optim.param_groups(model, loss, 1e-3, 0.0), lr=1e-3)
        e_rng = EMA(model, 0.9)
        checkpoint.save_checkpoint(os.path.join(tmp, "rng.pt"), model, loss, o, e_rng, 1, cfg)
        want = (torch.rand(()).item(), np.random.rand())
        checkpoint.load_checkpoint(os.path.join(tmp, "rng.pt"), model, loss, o, e_rng)
        check((torch.rand(()).item(), np.random.rand()) == want, "checkpoint: the torch and numpy random streams go on where they were saved")
        cuda = {a: getattr(torch.cuda, a) for a in ("is_available", "is_initialized", "device_count", "get_rng_state_all", "set_rng_state")}       # the CUDA generators, stubbed: no GPU is started here
        gpu_state, seen, asked = torch.arange(16, dtype=torch.uint8), [], []
        try:
            torch.cuda.is_available, torch.cuda.is_initialized = (lambda: True), (lambda: False)                # a GPU is there, this process has not started CUDA
            torch.cuda.get_rng_state_all = lambda: asked.append(1) or [gpu_state]
            checkpoint.save_checkpoint(os.path.join(tmp, "nocuda.pt"), model, loss, o, e_rng, 1, cfg)
            check("cuda" not in torch.load(os.path.join(tmp, "nocuda.pt"), weights_only=False)["rng"] and not asked, "checkpoint: a process that has not started CUDA saves no CUDA state and does not start it (the generators are not asked)")
            torch.cuda.is_initialized, torch.cuda.device_count = (lambda: True), (lambda: 1)
            torch.cuda.set_rng_state = lambda state, device: seen.append((device, state))
            checkpoint.save_checkpoint(os.path.join(tmp, "cuda.pt"), model, loss, o, e_rng, 1, cfg)
            checkpoint.load_checkpoint(os.path.join(tmp, "cuda.pt"), model, loss, o, e_rng)
            check(len(seen) == 1 and seen[0][0] == 0 and torch.equal(seen[0][1], gpu_state), "checkpoint: the CUDA generator of every device is saved and restored (a random op on the GPU goes on where it was)")
            two = [gpu_state, gpu_state + 1]
            torch.cuda.get_rng_state_all, torch.cuda.device_count = (lambda: two), (lambda: 2)
            checkpoint.save_checkpoint(os.path.join(tmp, "cuda2.pt"), model, loss, o, e_rng, 1, cfg)
            seen.clear()
            checkpoint.load_checkpoint(os.path.join(tmp, "cuda2.pt"), model, loss, o, e_rng)
            check([d for d, _ in seen] == [0, 1] and torch.equal(seen[0][1], two[0]) and torch.equal(seen[1][1], two[1]), "checkpoint: two devices, each generator goes back to its own device")
            torch.cuda.device_count = lambda: 1
            seen.clear()
            checkpoint.load_checkpoint(os.path.join(tmp, "cuda2.pt"), model, loss, o, e_rng)
            check([d for d, _ in seen] == [0], "checkpoint: a machine with fewer devices than the checkpoint restores the ones it has")
            seen.clear()
            checkpoint.load_checkpoint(os.path.join(tmp, "nocuda.pt"), model, loss, o, e_rng)
            torch.cuda.is_initialized = lambda: False
            checkpoint.load_checkpoint(os.path.join(tmp, "cuda.pt"), model, loss, o, e_rng)                   # a process without CUDA
            check(seen == [], "checkpoint: one without the CUDA key loads, and one with it loads in a process without CUDA, without touching a generator")
        finally:
            for a, v in cuda.items():
                setattr(torch.cuda, a, v)
        # accumulation: backward of two micro-batches (accum 2) = the gradient of their mean
        a, b = toy_data(2)
        m1, l1 = Net(), Loss()
        for batch in (a, b):
            StepInfo(0, 0.0, 0, 2).backward(l1(m1.fc(batch["x"]), batch["y"]))
        m2, l2 = Net(), Loss()
        (0.5 * (l2(m2.fc(a["x"]), a["y"]) + l2(m2.fc(b["x"]), b["y"]))).backward()
        check(torch.allclose(m1.fc.weight.grad, m2.fc.weight.grad, atol=1e-7) and torch.allclose(l1.log_w.grad, l2.log_w.grad, atol=1e-7), "StepInfo.backward: accum micro-batches give the gradient of their mean")
        # a stop request at the 7th step boundary: checkpoint, STOPPED, exit 75; the resume finishes the run and the log has one row per step
        try:
            run(tmp, "stop", stop_check=stop_after(6))
            raise AssertionError("the stop request must end the run")
        except SystemExit as e:
            check(e.code == control.EXIT_STOPPED == 75, "stop: exit code 75, never 0")
        st = json.load(open(os.path.join(tmp, "stop", "STOPPED")))
        check(st["step"] == 6 and st["reason"] == "selftest" and os.path.exists(os.path.join(tmp, "stop", "ckpt_00000006.pt")), "stop: STOPPED names the step, the checkpoint is there")
        try:                                             # the stop checkpoint prunes the older ones too: a checkpoint every 2 steps, keep 1, stop at step 7
            run(tmp, "keep", stop_check=stop_after(7), save_every=2, keep=1)
        except SystemExit:
            pass
        check(sorted(f for f in os.listdir(os.path.join(tmp, "keep")) if f.startswith("ckpt_")) == ["ckpt_00000007.pt"], "stop: the checkpoint of the stop is the only one left with keep 1")
        events = []

        class Count(Hooks):
            def begin_step(self, n):
                events.append(("begin", n))

            def batch(self, item):
                events.append("batch")
        cfg2, _, _, _ = run(tmp, "stop", hooks=Count())
        check(os.path.exists(os.path.join(tmp, "stop", "STOPPED.old")) and not os.path.exists(os.path.join(tmp, "stop", "STOPPED")), "resume: the files of the earlier stop are moved aside")
        check(events[0] == ("begin", 7) and events.count("batch") == 12 and sum(isinstance(e, tuple) for e in events) == 6, "resume: from step 6 to 12, the hooks see 6 steps of 2 micro-batches")
        check(sorted(json.loads(ln)["step"] for ln in open(os.path.join(tmp, "stop", "log.jsonl"))) == [1, 4, 8, 12], "resume: one log row per step (the run was stopped after step 6 and resumed)")
        # a resume must be the same configuration: another number of steps is refused, another cadence is not
        ck = torch.load(os.path.join(tmp, "stop", "ckpt_00000012.pt"), weights_only=False)["cfg"]
        checkpoint.check_resume(ck, dataclasses.replace(cfg2, log_every=7, name="elsewhere"), "x")
        try:
            checkpoint.check_resume(ck, dataclasses.replace(cfg2, steps=20), "x")
            raise AssertionError("a changed `steps` must be refused")
        except SystemExit as e:
            check("steps" in str(e.code), "check_resume: another `steps` is refused (it would restart the cosine), another log_every is not")
        # too many skipped steps in a row end the run
        try:
            run(tmp, "bad", step_fn=lambda batch, info: ({}, False, None), max_bad_steps=3)
            raise AssertionError("must abort")
        except RuntimeError as e:
            check("too many non-finite" in str(e), "fit: max_bad_steps skipped steps in a row abort the run")
    # EpochLoader: pass e is the same whenever and wherever it is built; iter_from(e, p) is that pass without its first p batches (also through persistent workers); no global stream is drawn
    data = torch.utils.data.TensorDataset(torch.arange(10.0))
    flat = lambda it: [b[0].tolist() for b in it]                                                           # noqa: E731
    loader = EpochLoader(data, 3, 7, drop_last=True)
    pass0, pass1 = flat(loader), flat(loader)
    check(pass0 != pass1 and len(pass0) == len(pass1) == 3, "EpochLoader: every pass is another order of the data (10 samples in batches of 3, drop_last: 3 batches)")
    check(flat(EpochLoader(data, 3, 7, drop_last=True).iter_from(1, 0)) == pass1 and flat(EpochLoader(data, 3, 8, drop_last=True).iter_from(1, 0)) != pass1, "EpochLoader: pass 1 is a function of (seed, epoch), the same in a new loader, another with another seed")
    check(flat(EpochLoader(data, 3, 7, drop_last=True).iter_from(1, 0)) != flat(EpochLoader(data, 3, 8, drop_last=True).iter_from(0, 0)),
          "EpochLoader: pass 1 of seed 7 is not pass 0 of seed 8 (seeds are not shifted copies of one order)")
    state = torch.get_rng_state()
    skipped = flat(loader.iter_from(1, 1))
    check(skipped == pass1[1:] and torch.equal(state, torch.get_rng_state()), "EpochLoader.iter_from(1, 1): pass 1 without its first batch; the global random stream was not drawn")
    workers = EpochLoader(data, 3, 7, drop_last=True, num_workers=2, persistent_workers=True)
    check(flat(workers.iter_from(1, 1)) == pass1[1:] and flat(workers.iter_from(0, 0)) == pass0 and flat(workers.iter_from(0, 3)) == [], "EpochLoader: the same through persistent workers, also from the end of a pass (nothing left)")
    # a stop at step 12 + the resume is the uninterrupted run, bit for bit: the data order (a function of seed and epoch, the position saved), the random op of the step (the global stream)
    with tempfile.TemporaryDirectory() as tmp:
        ref = resume_toy(tmp, "ref")
        check(resume_toy(tmp, "stp", stop=12) is None, "resume: the toy is stopped after step 12 (two and a half passes over its 10 batches)")
        d = os.path.join(tmp, "stp")
        check(torch.load(os.path.join(d, "ckpt_00000012.pt"), weights_only=False)["extra"]["data"] == {"epoch": 2, "pos": 4, "n": 40, "batch_size": 4}, "checkpoint: the position in the data (pass 2, 4 batches taken) and what it is valid for (40 samples, batch size 4)")
        check(same_run(ref, resume_toy(tmp, "stp")), "resume: stop + resume gives the uninterrupted run's weights, EMA and logged losses, bit for bit")
        plain_ref = resume_toy(tmp, "plain_ref", plain=True)
        resume_toy(tmp, "plain_stp", plain=True, stop=12)
        check(not same_run(plain_ref, resume_toy(tmp, "plain_stp", plain=True)), "resume: control, a plain shuffling DataLoader (global random stream, no position) is not the uninterrupted run")
        resume_toy(tmp, "old", stop=12)
        path = os.path.join(tmp, "old", "ckpt_00000012.pt")
        ck = torch.load(path, weights_only=False)
        del ck["extra"]["data"]                                                                              # a checkpoint from before the position was saved
        torch.save(ck, path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            done = resume_toy(tmp, "old")
        check(done is not None and "no data position in the checkpoint" in buf.getvalue(), "resume: a checkpoint without the position still resumes (from the start of pass 0) and says so")
        for what, kw in (("a dataset of another length", dict(n=36)), ("another batch size", dict(batch_size=2))):
            resume_toy(tmp, "fit_" + what[:7], stop=12)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                done = resume_toy(tmp, "fit_" + what[:7], **kw)
            check(done is not None and "does not fit these data" in buf.getvalue() and "pass 2 starts again at its beginning" in buf.getvalue(), f"resume: {what} than the checkpoint's position was made for: said, the pass starts again")
        resume_toy(tmp, "xpl", stop=12)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            done = resume_toy(tmp, "xpl", plain=True)
        check(done is not None and "these data cannot continue a pass" in buf.getvalue(), "resume: a plain loader meets a checkpoint with a position: said, not silently ignored")
    # a NaN input writes NaN into the batch-norm statistics: the step is skipped, but the run must not go on with a poisoned model (EMA and checkpoints would carry NaN)
    with tempfile.TemporaryDirectory() as tmp:
        class BNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc, self.bn = nn.Linear(4, 4), nn.BatchNorm1d(4)
        bn_batches = toy_data(4)
        bn_batches[2] = {"x": torch.full((8, 4), float("nan")), "y": bn_batches[2]["y"]}
        bn_model, bn_loss = BNNet(), Loss()

        def bn_step(batch, info: StepInfo):
            l = bn_loss(bn_model.bn(bn_model.fc(batch["x"])).sum(1, keepdim=True), batch["y"])
            info.backward(l)
            return {"total": float(l.detach())}, bool(torch.isfinite(l)), None
        cfg_bn = Config(name="bnpoison", out_root=tmp, steps=6, accum=2, warmup=1, lr=0.05, device="cpu", log_every=1, save_every=100, eval_every=0)
        try:
            fit(cfg_bn, bn_model, bn_loss, bn_batches, bn_step, prepare_run_dir(cfg_bn), "cpu")
            raise AssertionError("a NaN forward must not leave NaN batch-norm statistics unnoticed")
        except RuntimeError as e:
            check("batch-norm" in str(e) and not torch.isfinite(bn_model.bn.running_mean).all(), "fit: a non-finite loss that poisoned the batch-norm statistics ends the run (fp32 too), it does not skip on")
    # CUDA asked for and not there: an error with the text run.py retries, never a silent CPU run
    from trainkit.util import resolve_device
    real_available = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        try:
            resolve_device("cuda")
            raise AssertionError("cuda without CUDA must raise")
        except RuntimeError as e:
            check("CUDA error: unknown error" in str(e) and resolve_device("cpu") == "cpu", "resolve_device: cuda without CUDA raises (with the retry text of run.py), cpu stays cpu")
        torch.cuda.is_available = lambda: True
        check(resolve_device("cuda") == "cuda", "resolve_device: cuda with CUDA is cuda")
    finally:
        torch.cuda.is_available = real_available
    print("trainkit selftest passed")


if __name__ == "__main__":
    main()

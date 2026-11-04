"""Self-test of the engine on a toy (CPU, a few seconds): known answers for the schedule, the EMA, the clip and step rules, the resume rules and the stop protocol, and the
example of README.md end to end (gradient reaches every parameter, accumulation, checkpoints, resume, stop, hooks).
Run: python -m trainkit.selftest    (from autoresearch/)"""
import dataclasses
import json
import math
import os
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trainkit import checkpoint, control, optim, schedule  # noqa: E402
from trainkit.config import Config  # noqa: E402
from trainkit.ema import EMA  # noqa: E402
from trainkit.loop import Hooks, StepInfo, fit  # noqa: E402
from trainkit.util import prepare_run_dir  # noqa: E402


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


def run(tmp, name="toy", batches=None, stop_check=None, hooks=None, step_fn=None, **kw):
    cfg = Config(name=name, out_root=tmp, steps=12, accum=2, warmup=2, lr=0.05, device="cpu", log_every=4, save_every=6, eval_every=0, **kw)
    model, loss = Net(), Loss()

    def default_step(batch, info: StepInfo):             # one micro-batch: forward, backward (info.backward divides by accum), the scalar terms, finite?
        l = loss(model.fc(batch["x"]), batch["y"])
        info.backward(l)
        return {"total": float(l.detach())}, bool(torch.isfinite(l)), None
    out = fit(cfg, model, loss, toy_data() if batches is None else batches, step_fn or default_step, prepare_run_dir(cfg), "cpu", stop_check=stop_check, hooks=hooks)
    return cfg, model, loss, out


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
        cfg, model, loss, (_, _, _, last) = run(tmp)
        d = os.path.join(tmp, "toy")
        rows = [json.loads(ln) for ln in open(os.path.join(d, "log.jsonl"))]
        check([r["step"] for r in rows] == [1, 4, 8, 12] and last["step"] == 12 and all(math.isfinite(r["total"]) for r in rows), "fit: rows at step 1 and every log_every, the last row is returned")
        ref_model, ref_loss = Net(), Loss()
        check(all(not torch.equal(a, b) for a, b in list(zip(model.parameters(), ref_model.parameters())) + list(zip(loss.parameters(), ref_loss.parameters()))), "fit: every parameter of the model AND of the loss moved (the gradient reaches all)")
        check(sorted(f for f in os.listdir(d) if f.endswith(".pt")) == ["ckpt_00000006.pt", "ckpt_00000012.pt"] and os.path.exists(os.path.join(d, "config.json")), "fit: a checkpoint every save_every steps, the config copy")
        # accumulation: backward of two micro-batches (accum 2) = the gradient of their mean
        a, b = toy_data(2)
        m1, l1 = Net(), Loss()
        for batch in (a, b):
            StepInfo(0, 0.0, 0, 2).backward(l1(m1.fc(batch["x"]), batch["y"]))
        m2, l2 = Net(), Loss()
        (0.5 * (l2(m2.fc(a["x"]), a["y"]) + l2(m2.fc(b["x"]), b["y"]))).backward()
        check(torch.allclose(m1.fc.weight.grad, m2.fc.weight.grad, atol=1e-7) and torch.allclose(l1.log_w.grad, l2.log_w.grad, atol=1e-7), "StepInfo.backward: accum micro-batches give the gradient of their mean")
        # a stop request at the 7th step boundary: checkpoint, STOPPED, exit 75; the resume finishes the run and the log has one row per step
        calls = [0]
        calls_step = lambda run_dir: "selftest" if (calls.__setitem__(0, calls[0] + 1) or calls[0]) > 6 else None      # noqa: E731
        try:
            run(tmp, "stop", stop_check=calls_step)
            raise AssertionError("the stop request must end the run")
        except SystemExit as e:
            check(e.code == control.EXIT_STOPPED == 75, "stop: exit code 75, never 0")
        st = json.load(open(os.path.join(tmp, "stop", "STOPPED")))
        check(st["step"] == 6 and st["reason"] == "selftest" and os.path.exists(os.path.join(tmp, "stop", "ckpt_00000006.pt")), "stop: STOPPED names the step, the checkpoint is there")
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
    print("trainkit selftest passed")


if __name__ == "__main__":
    main()

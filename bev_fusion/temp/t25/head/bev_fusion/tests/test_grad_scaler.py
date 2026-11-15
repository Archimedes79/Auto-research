"""amp = "fp16" with a GradScaler (autoresearch/trainkit: amp_context, make_scaler, optimizer_step, bad_steps_after, scaler_extra; training/train.py: run_clip(scaler=...)); CPU, seconds, toy network.

  1. amp_context: bf16 / fp16 autocast on CUDA, nothing on the CPU or for "off"; an unknown name is an error (it used to mean off silently)
  2. make_scaler: only for fp16, with the start scale of the config
  3. optimizer_step without a scaler does what the loop did inline (clip, step, or drop a non-finite gradient): bit for bit over 5 steps
  4. with a scaler of a power of two the scaling and unscaling are exact, so 5 scaled steps give the same weights and the same gradient norm as 5 plain steps; the probe hook sees
     the UNscaled gradients
  5. an overflow (inf / nan in a gradient) skips the step, drops the gradient, halves the scale; a gradient whose norm alone overflows (every element finite) is skipped too
  6. bad_steps_after: an overflow is not a bad step until the scale is below 1
  7. run_clip(scaler=...) scales the loss before the backward (the gradients are scale x the plain ones, exactly) and logs the unscaled terms
  8. the scaler state travels in the checkpoint's `extra`; amp and fp16_scale are not free on a resume
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_grad_scaler.py
"""
import sys

import torch
from torch import nn

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
import train as T  # noqa: E402
from trainkit import amp as tk_amp, optim as tk_optim  # noqa: E402
from toy_train import ToyLoss, ToyNet, same_tree  # noqa: E402


fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


Scaler = lambda scale: torch.amp.GradScaler("cpu", init_scale=scale)      # noqa: E731
X, Y = torch.randn(16, 6, generator=torch.Generator().manual_seed(1)), torch.randn(16, 3, generator=torch.Generator().manual_seed(2))


def toy():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
    return m, torch.optim.AdamW(m.parameters(), lr=1e-2), list(m.parameters())


def mse(m):
    return ((m(X) - Y) ** 2).mean()


same = lambda a, b: all(torch.equal(p, q) for p, q in zip(a, b))      # noqa: E731

# 1 amp_context
for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    c = T.amp_context(name, "cuda")
    check(c.device == "cuda" and c.fast_dtype == dt and c._enabled, f"amp_context({name!r}, cuda): autocast in {dt}")
check(not T.amp_context("fp16", "cpu")._enabled and not T.amp_context("off", "cuda")._enabled and not T.amp_context("bf16", "cpu")._enabled, "no autocast on the CPU or for off")
errs = 0
for bad in ("fp32", "bf16 ", "FP16", ""):
    try:
        T.amp_context(bad, "cuda")
    except ValueError:
        errs += 1
check(errs == 4, "an unknown amp name is a ValueError")

# 2 make_scaler
cfg = T.TrainConfig(amp="fp16", fp16_scale=256.0)
sc = tk_amp.make_scaler(cfg, "cpu")
check(isinstance(sc, torch.amp.GradScaler) and sc.get_scale() == 256.0, "make_scaler(fp16): a GradScaler with the start scale of the config")
check(tk_amp.make_scaler(T.TrainConfig(), "cuda") is None and tk_amp.make_scaler(T.TrainConfig(amp="bf16"), "cuda") is None, "no scaler for off and bf16")
check(T.TrainConfig().amp == "off" and T.TrainConfig().fp16_scale == 1024.0, "defaults: amp off, start scale 1024")

# 3 without a scaler = the inline code
m1, o1, p1 = toy()
m2, o2, p2 = toy()
ok3 = True
for i in range(5):
    for o in (o1, o2):
        o.zero_grad(set_to_none=True)
    mse(m1).backward()
    gn1 = torch.nn.utils.clip_grad_norm_(p1, 0.5)
    o1.step()
    mse(m2).backward()
    gn2, st = T.optimizer_step(o2, p2, 0.5)
    ok3 &= torch.equal(gn1, gn2) and st == "ok"
check(ok3 and same(p1, p2), "optimizer_step without a scaler: the same gradient norm and weights as clip + step inline, 5 steps")
m2b, o2b, p2b = toy()
before = [p.detach().clone() for p in p2b]
mse(m2b).backward()
p2b[0].grad[0, 0] = float("inf")
gn, st = T.optimizer_step(o2b, p2b, 0.5)
check(st == "nonfinite" and not torch.isfinite(gn) and same(before, p2b) and all(p.grad is None for p in p2b), "an inf gradient without a scaler: status nonfinite, no step, gradients dropped")

# 4 a scale of a power of two is exact
m3, o3, p3 = toy()
sc3 = Scaler(1024.0)
m4, o4, p4 = toy()
seen, plain_grads = [], []
ok4 = True
for i in range(5):
    for o in (o3, o4):
        o.zero_grad(set_to_none=True)
    sc3.scale(mse(m3)).backward()
    mse(m4).backward()
    plain_grads.append([p.grad.clone() for p in p4])
    gn3, st3 = T.optimizer_step(o3, p3, 0.5, sc3, before_clip=lambda: seen.append([p.grad.clone() for p in p3]))
    gn4, st4 = T.optimizer_step(o4, p4, 0.5)
    ok4 &= torch.equal(gn3, gn4) and st3 == st4 == "ok"
check(ok4 and same(p3, p4), "5 steps with a scaler of 1024: the same weights and gradient norms as 5 plain steps, bit for bit")
check(all(same(s, g) for s, g in zip(seen, plain_grads)) and len(seen) == 5, "the hook before the clip (the probe) sees the unscaled gradients")
check(sc3.get_scale() == 1024.0, "no overflow, no growth yet: the scale stays 1024")

# 5 overflow
m5, o5, p5 = toy()
sc5 = Scaler(1024.0)
w0 = [p.detach().clone() for p in p5]
log5 = []
for i, poison in enumerate((float("inf"), float("nan"), None)):
    o5.zero_grad(set_to_none=True)
    sc5.scale(mse(m5)).backward()
    if poison is not None:
        p5[0].grad[0, 0] = poison
    s0 = sc5.get_scale()
    gn, st = T.optimizer_step(o5, p5, 0.5, sc5)
    log5.append((st, s0, sc5.get_scale(), same(w0, p5)))
check([x[0] for x in log5] == ["overflow", "overflow", "ok"] and [x[2] for x in log5] == [512.0, 256.0, 256.0], f"inf, nan: overflow and the scale halves each time (1024 -> 512 -> 256); then a clean step ({[x[0] for x in log5]})")
check(log5[0][3] and log5[1][3] and not log5[2][3], "weights unchanged by the overflows, moved by the clean step")
m6, o6, p6 = toy()
sc6 = Scaler(1024.0)
w6 = [p.detach().clone() for p in p6]
sc6.scale(mse(m6)).backward()
for p in p6:
    p.grad.fill_(3.0e38)                                                    # every element finite, but the norm overflows float32
gn, st = T.optimizer_step(o6, p6, 0.5, sc6)
check(st == "overflow" and not torch.isfinite(gn) and same(w6, p6) and sc6.get_scale() == 512.0,
      "finite elements whose norm overflows: skipped as an overflow, no step (the scaler alone would have stepped), scale 1024 -> 512")

# 6 the count of bad steps
sc7 = Scaler(1024.0)
low = Scaler(0.5)
check(tk_optim.bad_steps_after("ok", 3, sc7) == 0 and tk_optim.bad_steps_after("nonfinite", 3, sc7) == 4 and tk_optim.bad_steps_after("nonfinite", 0, None) == 1, "ok resets, a non-finite loss or gradient counts")
check(tk_optim.bad_steps_after("overflow", 3, sc7) == 3 and tk_optim.bad_steps_after("overflow", 3, low) == 4, "an fp16 overflow counts only once the scale is below 1")

# 7 run_clip with a scaler
w = nn.Parameter(torch.tensor([0.7, -0.4, 1.3]))


def fake_forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    tot = (w * batch["x"]).sum()
    return {"total": tot, "group.areas": (w ** 2).sum()}, {"state": {"m": w.detach() * 2}}, None


keep = T.forward_loss, T.clip_step
T.forward_loss, T.clip_step = fake_forward_loss, (lambda prev, device: (0.5, 1.0, 0.0))
try:
    res = {}
    for label, scaler in (("plain", None), ("scaled", Scaler(1024.0))):
        w.grad = None
        clip = [{"x": torch.tensor([1.0 + t, 2.0, -1.0]), "tokens": ["a"]} for t in range(3)]
        r = T.run_clip(None, None, clip, 0.3, "cpu", "off", T.TrainConfig(accum=2, seq_bptt=True, flip_aug=0.0), backward=True, scaler=scaler)
        res[label] = (r, w.grad.clone())
finally:
    T.forward_loss, T.clip_step = keep
(rp, gp), (rs, gs) = res["plain"], res["scaled"]
check(rp[0] == rs[0] and rp[1] and rs[1] and rp[3] == rs[3], "run_clip(scaler): the logged terms and totals are the unscaled ones, the same as without")
check(torch.equal(gs, gp * 1024.0), "run_clip(scaler): the gradients are exactly 1024 x the plain ones")

# 8 the checkpoint
sc8 = Scaler(1024.0)
sc8.scale(torch.tensor(1.0))                                                # the scale is created lazily, by the first scale() (train() always scales before it updates)
sc8.update(64.0)
ex = tk_amp.scaler_extra(sc8)
sc9 = Scaler(1024.0)
sc9.load_state_dict(ex["scaler"])
check(tk_amp.scaler_extra(None) == {} and sc9.get_scale() == 64.0, "the scale travels in the checkpoint's extra; no scaler, nothing")
check("amp" not in T.RESUME_FREE and "fp16_scale" not in T.RESUME_FREE, "amp and fp16_scale are not free on a resume")
old = {"amp": "off", "fp16_scale": 1024.0}
try:
    T.check_resume(old, T.TrainConfig(amp="fp16"), "toy")
    refused = False
except SystemExit:
    refused = True
check(refused, "a resume that switches amp from off to fp16 is refused")

# 9 the growth counter, the boundary of the bad-step rule, the config knob
scg = Scaler(1024.0)
scg = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=4)
mg, og, pg = toy()
seq9 = []
for poison in (None, None, None, float("inf"), None):                     # three clean steps, an overflow, a clean step: an overflow restarts the count of clean steps
    og.zero_grad(set_to_none=True)
    scg.scale(mse(mg)).backward()
    if poison is not None:
        pg[0].grad[0, 0] = poison
    seq9.append(T.optimizer_step(og, pg, 0.5, scg)[1])
check(seq9 == ["ok", "ok", "ok", "overflow", "ok"] and scg.get_scale() == 512.0, f"an overflow restarts the count of clean steps: scale 1024 -> 512 and it does not double again at once ({scg.get_scale()})")
check(tk_optim.bad_steps_after("overflow", 3, Scaler(1.0)) == 3 and tk_optim.bad_steps_after("overflow", 3, Scaler(0.9999)) == 4, "the boundary: at scale exactly 1 an overflow still does not count")
check(tk_amp.make_scaler(T.TrainConfig(amp="fp16", fp16_growth_interval=7), "cpu").get_growth_interval() == 7, "fp16_growth_interval reaches the scaler")
check(tk_optim.poisoned_buffers(nn.Sequential(nn.BatchNorm1d(3))) == [] and tk_optim.poisoned_buffers(nn.Sequential(nn.BatchNorm1d(3)).apply(lambda m: m.running_mean.fill_(float("inf")) if hasattr(m, "running_mean") else None)) == ["0.running_mean"],
      "poisoned_buffers names the batch-norm statistics that are not finite")

# 10 the REAL train() on a CPU toy (the data path stubbed): fp16 with a power-of-two scale = amp off, bit for bit, also through overflows; the scale in rows and checkpoint
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402


class Poison(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.f, None


DIAGS = []


def toy_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    DIAGS.append(bool(diag))
    mode = batch["mode"]
    x = batch["x"] * float("inf") if mode == "bn_poison" else batch["x"]
    h = model.net[2](model.net[1](model.net[0](x)))
    if mode == "inf_grad":
        h = Poison.apply(h, float("inf"))
    tot = ((model.net[3](h) - batch["y"]) ** 2).mean() + 0.01 * loss.s.pow(2).sum()
    if mode in ("nan_loss", "bn_poison"):
        tot = tot * float("nan")
    return {"total": tot, "group.a": tot.detach()}, {"state": {"m": h.detach()}}, None


def toy_batches(modes, n=40):
    g = torch.Generator().manual_seed(7)
    return [{"x": torch.randn(16, 6, generator=g), "y": torch.randn(16, 3, generator=g), "tokens": [f"t{i}"], "mode": modes.get(i, "ok")} for i in range(n)]


saved = {k: getattr(T, k) for k in ("build", "forward_loss", "make_loader", "val_subset")}


def patch(forward=toy_forward):
    T.build = lambda cfg: (ToyNet(), ToyLoss(), True)
    T.forward_loss = forward
    T.make_loader = lambda ds, cfg, train, workers=None, generator=None: ds
    T.val_subset = lambda ds, n: ds


def run_train(root, name, modes, crash_at=None, **kw):
    patch()
    base = dict(name=name, out_root=root, steps=12, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, max_bad_steps=3,
                flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5)
    base.update(kw)
    cfg = T.TrainConfig(**base)
    data = toy_batches(modes)
    if crash_at is not None:
        n = {"k": 0}

        def crashing(*a, **k):
            n["k"] += 1
            if n["k"] == crash_at:
                raise KeyboardInterrupt("simulated crash")
            return toy_forward(*a, **k)
        T.forward_loss = crashing
    buf, exc = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            T.train(cfg, datasets=(data, data))
    except BaseException as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
    d = os.path.join(root, name)
    rows = [json.loads(ln) for ln in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    cks = sorted(f for f in os.listdir(d) if f.startswith("ckpt_") and f.endswith(".pt")) if os.path.isdir(d) else []
    ck = torch.load(os.path.join(d, cks[-1]), map_location="cpu", weights_only=False) if cks else None
    return dict(out=buf.getvalue(), rows=rows, ck=ck, exc=exc)


unscaled = lambda rows: [{k: v for k, v in r.items() if k not in ("t", "s_per_step", "loss_scale", "fp16_overflows")} for r in rows]      # noqa: E731
try:
    root = tempfile.mkdtemp(prefix="gs_")
    off, fp = run_train(root, "off", {}), run_train(root, "fp16", {}, amp="fp16")
    check(off["exc"] is None and fp["exc"] is None and same_tree(off["ck"]["state"], fp["ck"]["state"]) and same_tree(off["ck"]["ema"], fp["ck"]["ema"]) and same_tree(off["ck"]["opt"], fp["ck"]["opt"]),
          "train(): amp fp16 (scale 1024, CPU) gives the same weights, EMA and optimiser state as amp off over a whole run, bit for bit")
    check(unscaled(off["rows"]) == unscaled(fp["rows"]) and all(r["loss_scale"] == 1024.0 and r["fp16_overflows"] == 0 for r in fp["rows"]) and "loss_scale" not in off["rows"][0],
          "the rows are the same, plus loss_scale / fp16_overflows only under fp16")
    check(fp["ck"]["extra"]["scaler"]["scale"] == 1024.0 and off["ck"]["extra"] == {}, "the checkpoint holds the scaler state under fp16, nothing under off")
    mods = {3: "inf_grad", 9: "nan_loss"}                                   # an overflow in step 2, a non-finite loss in step 5
    off2, fp2 = run_train(root, "off2", mods), run_train(root, "fp16b", mods, amp="fp16")
    check(off2["exc"] is None and fp2["exc"] is None and same_tree(off2["ck"]["state"], fp2["ck"]["state"]) and same_tree(off2["ck"]["ema"], fp2["ck"]["ema"]) and same_tree(off2["ck"]["opt"], fp2["ck"]["opt"]),
          "with an overflow and a non-finite loss in the run: still the same weights, EMA and optimiser state as amp off (the skipped steps are skipped alike)")
    last = fp2["rows"][-1]
    check(last["loss_scale"] == 512.0 and last["fp16_overflows"] == 1 and fp2["ck"]["extra"]["scaler"]["scale"] == 512.0 and "fp16 gradient overflow, skipped, loss scale -> 512" in fp2["out"],
          "the overflow halved the scale once (1024 -> 512), the non-finite loss left it alone; rows, checkpoint and the printed line say so")
    resumed = run_train(root, "fp16b", {}, amp="fp16")                                 # the same run again, clean data: it resumes from the last checkpoint, with the scale it had reached
    check("resumed from" in resumed["out"] and resumed["ck"]["extra"]["scaler"]["scale"] == 512.0, "a resume restores the scale from the checkpoint (512, not the start scale 1024)")
    crash = run_train(root, "crash", {3: "inf_grad"}, amp="fp16", crash_at=15)
    again = run_train(root, "crash", {}, amp="fp16")
    check(crash["exc"] and "KeyboardInterrupt" in crash["exc"] and "resumed from" in again["out"] and again["ck"]["extra"]["scaler"]["scale"] == 512.0, "a crash after an overflow, then a rerun: the scale 512 is restored")
    loop = run_train(root, "spiral", {i: "inf_grad" for i in range(40)}, amp="fp16", fp16_scale=2.0)
    check(loop["exc"] and "too many non-finite steps in a row" in loop["exc"] and "non-finite gradient (fp16 overflow, loss scale 0.125), skipped (3 in a row)" in loop["out"],
          "overflow after overflow: from scale 0.5 on it counts, three in a row end the run, in the words the health check counts")
    pois = run_train(root, "bn", {2: "bn_poison"}, amp="fp16")
    check(pois["exc"] and "RuntimeError" in pois["exc"] and "batch-norm statistics" in pois["exc"], "fp16: a forward that left non-finite batch-norm statistics stops the run (no loss scale cures it) instead of poisoning the EMA")
    check(run_train(root, "bad", {}, amp="FP16")["exc"].startswith("ValueError"), "an unknown amp name stops train() before anything is built")
    DIAGS.clear()
    check("grad_diag_every is ignored" in run_train(root, "diag", {}, amp="fp16", grad_diag_every=3)["out"] and not any(DIAGS), "fp16 with grad_diag_every: ignored, with a warning, and no step asks for the diag")
    DIAGS.clear()
    run_train(root, "diag_off", {}, grad_diag_every=3)
    check(any(DIAGS), "(control) amp off with grad_diag_every 3 does ask for the diag")
    calls = {"n": 0}
    keep_stop = T._stop_requested
    T._stop_requested = lambda d: "test" if (calls.__setitem__("n", calls["n"] + 1) or calls["n"]) > 6 else None          # a stop request at the 7th step (the Training stop protocol)
    stopped = run_train(root, "stopped", {3: "inf_grad"}, amp="fp16", steps=40, save_every=0)
    T._stop_requested = keep_stop
    check(stopped["exc"] and "SystemExit: 75" in stopped["exc"] and stopped["ck"]["extra"]["scaler"]["scale"] == 512.0, "a stop request (exit 75) saves the checkpoint with the scaler state")
finally:
    for k, v in saved.items():
        setattr(T, k, v)

print("GRAD SCALER GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)

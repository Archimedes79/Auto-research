"""The split gradient clip of training/train.py (`TrainConfig.grad_clip_split`; CPU, a few seconds, toy modules, no data):

  1. without a split `clip_gradients` is the one global torch call: the same gradients bit for bit, the same returned norm
  2. with a split each listed module and the rest are clipped on their own to grad_clip: a block with a 100 x larger gradient does not scale the others down (the t16 finding:
     the motion block's gradient norm 36 - 131 against 0.1 - 17 for every other module made the global clip factor 0.04 - 0.1); a group below the limit is left alone
  3. the returned norm is that of ALL gradients before the clip (what the log and the probe report), also with a split; a non-finite gradient in any group makes it non-finite
  4. `clip_groups`: a prefix matches the module and its children, never a longer name that starts the same ("big" is not "bigger"); a prefix that names no trainable parameter is an error
  5. `optimizer_step` with a split makes the step and reports "ok" / "nonfinite" as without; the config field is a TrainConfig field, off by default, and not free on a resume
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_clip_split.py
"""
import dataclasses
import math
import sys

import torch
from torch import nn

sys.path.insert(0, "training")
import train as T  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.big = nn.Linear(4, 4)
        self.bigger = nn.Linear(4, 4)        # a name that starts like "big"
        self.small = nn.Linear(4, 4)


def with_grads(norms):
    """a toy model whose gradient norms are exactly `norms` = {module: norm} (direction fixed, scaled)"""
    m = Toy()
    torch.manual_seed(0)
    for name, nrm in norms.items():
        ps = list(getattr(m, name).parameters())
        for p in ps:
            p.grad = torch.randn_like(p)
        tot = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps))
        for p in ps:
            p.grad *= nrm / tot
    return m


def gnorm(ps):
    return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps))


# 1 without a split: the global torch call
a, b = with_grads({"big": 100.0, "bigger": 3.0, "small": 4.0}), with_grads({"big": 100.0, "bigger": 3.0, "small": 4.0})
ra = T.clip_gradients(list(a.parameters()), 5.0)
rb = torch.nn.utils.clip_grad_norm_(list(b.parameters()), 5.0)
check(torch.equal(ra, rb) and all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters())), "no split: the same bits as the one global torch call")
check(abs(float(ra) - math.sqrt(100 ** 2 + 3 ** 2 + 4 ** 2)) < 1e-4, f"no split: the returned norm is the global norm before the clip ({float(ra):.4f})")

# 2 with a split
m = with_grads({"big": 100.0, "bigger": 12.0, "small": 3.0})
split = T.clip_groups(m, ("big",))
r = T.clip_gradients(list(m.parameters()), 5.0, split)
check(abs(gnorm(m.big.parameters()) - 5.0) < 1e-4, f"split: the listed module is clipped to 5 on its own ({gnorm(m.big.parameters()):.4f})")
check(abs(gnorm(list(m.bigger.parameters()) + list(m.small.parameters())) - 5.0) < 1e-4,
      "split: the rest is clipped to 5 as one group (12 and 3 together are 12.4)")
m2 = with_grads({"big": 100.0, "bigger": 2.0, "small": 1.0})
T.clip_gradients(list(m2.parameters()), 5.0, T.clip_groups(m2, ("big",)))
check(abs(gnorm(m2.bigger.parameters()) - 2.0) < 1e-5 and abs(gnorm(m2.small.parameters()) - 1.0) < 1e-5,
      "split: a group below the limit is left alone (the global clip would have scaled it by 0.05)")
m3 = with_grads({"big": 100.0, "bigger": 2.0, "small": 1.0})
T.clip_gradients(list(m3.parameters()), 5.0)
check(gnorm(m3.small.parameters()) < 0.06, "control: the global clip does scale the small module down to 0.05 (the problem the split removes)")

# 3 the returned norm
check(abs(float(r) - math.sqrt(100 ** 2 + 12 ** 2 + 3 ** 2)) < 1e-3, f"split: the returned norm is that of all gradients before the clip ({float(r):.3f})")
m4 = with_grads({"big": 1.0, "bigger": 1.0, "small": 1.0})
m4.small.weight.grad[0, 0] = float("nan")
check(not bool(torch.isfinite(T.clip_gradients(list(m4.parameters()), 5.0, T.clip_groups(m4, ("big",))))), "split: a non-finite gradient in the rest makes the returned norm non-finite")
m5 = with_grads({"big": 1.0, "bigger": 1.0, "small": 1.0})
m5.big.bias.grad[0] = float("inf")
check(not bool(torch.isfinite(T.clip_gradients(list(m5.parameters()), 5.0, T.clip_groups(m5, ("big",))))), "split: a non-finite gradient in the listed module makes it non-finite")

# 4 clip_groups
g = T.clip_groups(Toy(), ("big",))
check(len(g) == 1 and len(g[0]) == 2, "clip_groups: the prefix 'big' takes the weight and bias of big, not of bigger")
check(len(T.clip_groups(Toy(), ("big.weight",))[0]) == 1, "clip_groups: a parameter name is a prefix too")
try:
    T.clip_groups(Toy(), ("nothing",))
    check(False, "clip_groups: an unknown prefix is an error")
except ValueError as e:
    check("nothing" in str(e), "clip_groups: an unknown prefix is an error that names it")
frozen = Toy()
for p in frozen.big.parameters():
    p.requires_grad_(False)
try:
    T.clip_groups(frozen, ("big",))
    check(False, "clip_groups: a frozen module is no trainable parameter")
except ValueError:
    check(True, "clip_groups: a frozen module is no trainable parameter (an error)")

# 5 optimizer_step and the config
m6 = with_grads({"big": 100.0, "bigger": 3.0, "small": 3.0})
opt = torch.optim.SGD(m6.parameters(), lr=0.1)
w0 = m6.small.weight.detach().clone()
gn, status = T.optimizer_step(opt, list(m6.parameters()), 5.0, split=T.clip_groups(m6, ("big",)))
check(status == "ok" and abs(float(gn) - math.sqrt(100 ** 2 + 3 ** 2 + 3 ** 2)) < 1e-3 and not torch.equal(w0, m6.small.weight), "optimizer_step with a split: the step is made, the norm is the total before the clip")
m7 = with_grads({"big": 1.0, "bigger": 1.0, "small": 1.0})
m7.big.weight.grad[0, 0] = float("nan")
_, status = T.optimizer_step(torch.optim.SGD(m7.parameters(), lr=0.1), list(m7.parameters()), 5.0, split=T.clip_groups(m7, ("big",)))
check(status == "nonfinite", "optimizer_step with a split: a non-finite gradient is no step")
fields = {f.name: f for f in dataclasses.fields(T.TrainConfig)}
check("grad_clip_split" in fields and fields["grad_clip_split"].default == (), "TrainConfig.grad_clip_split exists and is off by default")
check("grad_clip_split" not in T.RESUME_FREE, "grad_clip_split is not free on a resume (it changes the update)")

# 6 the critic's gaps (2026-11-02): wiring, two prefixes, groups without gradients, the scaler, resume, overlaps
m8 = with_grads({"big": 100.0, "bigger": 20.0, "small": 3.0})
lr = 0.1
w_small, w_big, w_bigger = m8.small.weight.detach().clone(), m8.big.weight.detach().clone(), m8.bigger.weight.detach().clone()
g_small, g_big, g_bigger = m8.small.weight.grad.clone(), m8.big.weight.grad.clone(), m8.bigger.weight.grad.clone()
T.optimizer_step(torch.optim.SGD(m8.parameters(), lr=lr), list(m8.parameters()), 5.0, split=T.clip_groups(m8, ("big", "bigger")))
sc_big, sc_bigger = 5.0 / 100.0, 5.0 / 20.0                    # each listed module is its own group (norms 100 and 20); the rest (small, norm 3) is below the limit
check(torch.allclose(m8.small.weight, w_small - lr * g_small, atol=1e-6), "wiring: the rest (below the limit) moves by lr * grad, not by lr * grad * 0.05 (the global clip)")
check(torch.allclose(m8.big.weight, w_big - lr * sc_big * g_big, atol=1e-6) and torch.allclose(m8.bigger.weight, w_bigger - lr * sc_bigger * g_bigger, atol=1e-6),
      "two prefixes: each listed module moves by its own clip factor (0.05 and 0.25)")
m9 = with_grads({"big": 100.0, "bigger": 1.0, "small": 1.0})
for p in list(m9.bigger.parameters()) + list(m9.small.parameters()):
    p.grad = None                                              # the rest has no gradient in this step (a frozen-in-this-step branch): no crash, the norm is the listed module's
r9 = T.clip_gradients(list(m9.parameters()), 5.0, T.clip_groups(m9, ("big",)))
check(abs(float(r9) - 100.0) < 1e-3 and abs(gnorm(m9.big.parameters()) - 5.0) < 1e-4, "a rest without any gradient: no crash, the returned norm is the listed module's")
m10 = with_grads({"big": 100.0, "bigger": 2.0, "small": 1.0})
for p in m10.big.parameters():
    p.grad = None
r10 = T.clip_gradients(list(m10.parameters()), 5.0, T.clip_groups(m10, ("big",)))
check(abs(float(r10) - math.sqrt(5.0)) < 1e-3, "a listed module without any gradient: its group counts 0, the rest is clipped on its own")
m11 = with_grads({"big": 100.0, "bigger": 2.0, "small": 1.0})
check(len(T.clip_groups(m11, ("big", "bigger", "small"))) == 3 and T.clip_gradients(list(m11.parameters()), 5.0, T.clip_groups(m11, ("big", "bigger", "small"))) is not None,
      "every module listed: the rest is empty and that is fine")
m12 = with_grads({"big": 1.0, "bigger": 1.0, "small": 1.0})
m12.small.weight.grad[0, 0] = float("nan")
scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
scaler.scale(torch.ones(1))                                    # the scale is created lazily by the first scale() call
_, status = T.optimizer_step(torch.optim.SGD(m12.parameters(), lr=0.1), list(m12.parameters()), 5.0, scaler=scaler, split=T.clip_groups(m12, ("big",)))
check(status == "overflow" and scaler.get_scale() == 512.0, f"with an fp16 scaler a NaN in the rest is an overflow (status {status}, scale {scaler.get_scale()})")
for bad_split, why in ((("big", "big.weight"), "overlap"), ("big", "a bare string")):
    try:
        T.clip_groups(Toy(), bad_split)
        check(False, f"clip_groups: {why} is an error")
    except ValueError as e:
        check(True, f"clip_groups: {why} is an error ({str(e)[:60]}...)")
old = dataclasses.asdict(T.TrainConfig(name="x"))
old.pop("grad_clip_split")                                     # a checkpoint from before the switch
try:
    T.check_resume(old, T.TrainConfig(name="x", grad_clip_split=("motion_block",)), "x")
    check(False, "resume: an old checkpoint (no key) with the split turned on is refused")
except SystemExit as e:
    check("grad_clip_split" in str(e), "resume: an old checkpoint (no key) with the split turned on is refused")
T.check_resume(old, T.TrainConfig(name="x"), "x")
check(True, "resume: an old checkpoint with the split off resumes")

# 7 the REAL train() call site on a toy (data path stubbed): the split changes the weights as the arithmetic says, a typo fails before the data are built
import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402


class TwoNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.big = nn.Linear(6, 3)
        self.small = nn.Linear(6, 3)
        self.grid = None

    def forward(self, batch, dts=None):
        return self.big(batch["x"]), self.small(batch["x"])


class ToyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))


def toy_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    a, b = model(batch)
    tot = 100.0 * ((a - batch["y"]) ** 2).mean() + ((b - batch["y"]) ** 2).mean() + 0.01 * loss.s.pow(2).sum()
    return {"total": tot, "group.a": tot.detach()}, {"state": {"m": a.detach()}}, None


def run_toy(root, name, **kw):
    saved = {k: getattr(T, k) for k in ("build", "forward_loss", "make_loader", "val_subset")}
    T.build = lambda cfg: (TwoNet(), ToyLoss(), True)
    T.forward_loss = toy_forward
    T.make_loader = lambda ds, cfg, train, workers=None, generator=None: ds
    T.val_subset = lambda ds, n: ds
    g = torch.Generator().manual_seed(7)
    data = [{"x": torch.randn(16, 6, generator=g), "y": 3 * torch.randn(16, 3, generator=g), "tokens": [f"t{i}"]} for i in range(8)]
    cfg = T.TrainConfig(name=name, out_root=root, steps=6, accum=1, warmup=1, log_every=1, eval_every=0, save_every=6, keep=1, device="cpu", workers=0,
                        flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=1.0, **kw)
    exc = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T.train(cfg, datasets=(data, data))
    except BaseException as e:  # noqa: BLE001
        exc = e
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    d = os.path.join(root, name)
    rows = [json.loads(ln) for ln in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    cks = sorted(f for f in os.listdir(d) if f.startswith("ckpt_")) if os.path.isdir(d) else []
    return exc, rows, (torch.load(os.path.join(d, cks[-1]), map_location="cpu", weights_only=False) if cks else None)


import json  # noqa: E402
with tempfile.TemporaryDirectory() as root:
    e0, rows0, ck0 = run_toy(root, "nosplit")
    e1, rows1, ck1 = run_toy(root, "split", grad_clip_split=("big",))
    check(e0 is None and e1 is None, f"train() runs with and without the split ({e0}, {e1})")
    if ck0 and ck1:
        w0 = {k: v for k, v in ck0["state"].items() if k.startswith("model.small")}
        w1 = {k: v for k, v in ck1["state"].items() if k.startswith("model.small")}
        check(bool(w0) and any(not torch.equal(w0[k], w1[k]) for k in w0), "train(): the split changes the weights of the unlisted module (the call site is wired)")
    r0, r1 = rows0[0], rows1[0]
    check(abs(r0["grad_norm"] - r1["grad_norm"]) < 1e-4 * max(1.0, r0["grad_norm"]), f"train(): the logged grad_norm of step 1 is the same global norm before the clip ({r0['grad_norm']:.4f} / {r1['grad_norm']:.4f})")
    e2, _, _ = run_toy(root, "typo", grad_clip_split=("nothing",))
    check(isinstance(e2, ValueError) and "nothing" in str(e2), "train(): a typo in a prefix raises ValueError (before any data is read)")

print("test_clip_split " + ("FAILED: " + "; ".join(fails) if fails else f"passed"))
sys.exit(1 if fails else 0)

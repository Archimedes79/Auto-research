"""The speed changes of training/train.py and probe.py that need no training run to be tested (CPU, a few seconds, toy data):

  1. run_clip reads its scalar terms back with ONE transfer (`_pack` / `_unpack`) and the result is bit-identical to the old float() per term: same values, same order of
     the sums, same key order, the totals per keyframe; a term that is a view of a parameter which a later keyframe changes in place (the Kendall weights) is read as it was
     when its keyframe was packed
  2. the opt-in switches are off by default and then nothing is touched (apply_speed_flags returns {}, the loader keeps the torch defaults); on, they set what they say
  3. to_device accepts non_blocking and changes nothing else
  4. the new fields are in the right place for a resume: the ones that change only speed are free, the ones that change numbers are not
  5. StepProfiler writes profile.txt (window, synchronisations per step, operators) after warm-up + window steps, also when the run ends early, once, and not at all when off
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_speed_flags.py
"""
import dataclasses
import os
import random
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, "training")
import probe  # noqa: E402
import train as T  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1 one transfer, the same numbers
selected = lambda k: k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in T.LOGGED_TERMS      # noqa: E731
rng = random.Random(0)
kendall = nn.Parameter(torch.tensor([0.3, -0.2]))                                  # a Parameter that a later keyframe changes in place


def keyframe(i):
    return {"total": torch.tensor(rng.uniform(5, 15)), "group.areas": torch.tensor(rng.random()), "group.lines": torch.tensor(rng.random() * 3),
            "s.areas": kendall[0].detach() if False else kendall[0], "consistency": 0.25 * i, "radar_aux": torch.tensor(rng.random()).bfloat16(),
            "objhead.vel@+0.0": torch.tensor([rng.random()]), "not_logged": torch.tensor(99.0), "diag": {"a": 1}}


L, outs = 3, []
kf, old_acc, old_tot = [], {}, []
for i in range(L):
    out = keyframe(i)
    outs.append(out)
    kf.append(T._pack(out, selected))                                              # the new code packs here ...
    old_tot.append(T._f(out["total"]))                                             # ... the old code read here, right after the keyframe
    for k, v in out.items():
        if selected(k):
            old_acc[k] = old_acc.get(k, 0.0) + T._f(v) / L
    with torch.no_grad():
        kendall.clamp_(-0.25, 0.25)                                                # the next keyframe's forward changes the weights in place
        kendall.mul_(0.9)
acc, tot = T._unpack(kf, L)
check(acc == old_acc and list(acc) == list(old_acc) and tot == old_tot, f"bit-identical to the float() per term: {len(acc)} terms, {len(tot)} totals, same key order")
check("not_logged" not in acc and "diag" not in acc and acc["consistency"] == (0.0 + 0.25 + 0.5) / L, "only the logged terms; numbers are kept as they are")
check(T._unpack([], 3) == ({}, []), "nothing packed, nothing read")
calls = []
real_cpu = torch.Tensor.cpu
torch.Tensor.cpu = lambda self, *a, **k: (calls.append(1), real_cpu(self, *a, **k))[1]
try:
    T._unpack(kf, L)
finally:
    torch.Tensor.cpu = real_cpu
check(len(calls) == 1, f"ONE device-to-host transfer for the whole clip (not one per term): {len(calls)}")

# 2 the switches
class Toy(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return i


old_bench, old_prec, old_tf32 = torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
try:
    check(T.apply_speed_flags(T.TrainConfig()) == {} and torch.backends.cudnn.benchmark == old_bench and torch.get_float32_matmul_precision() == old_prec,
          "defaults: nothing is set, nothing is touched")
    got = T.apply_speed_flags(T.TrainConfig(cudnn_benchmark=True, tf32=True))
    check(got == {"cudnn.benchmark": True, "tf32": "high"} and torch.backends.cudnn.benchmark is True and torch.get_float32_matmul_precision() == "high", f"on: {got}")
finally:
    torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32 = old_bench, old_tf32
    torch.set_float32_matmul_precision(old_prec)
base = T.make_loader(Toy(), T.TrainConfig(workers=0, batch_size=2), True)
check(base.pin_memory is False and base.prefetch_factor is None, "loader defaults: not pinned, torch's own prefetch")
pin = T.make_loader(Toy(), T.TrainConfig(workers=0, batch_size=2, pin_memory=True), True)
check(pin.pin_memory == torch.cuda.is_available(), f"pin_memory only where there is a GPU to copy to ({torch.cuda.is_available()})")
pre = T.make_loader(Toy(), T.TrainConfig(workers=2, batch_size=2, prefetch_factor=4), True)
check(pre.prefetch_factor == 4 and T.make_loader(Toy(), T.TrainConfig(workers=0, prefetch_factor=4), True).prefetch_factor is None, "prefetch_factor with workers; ignored without")

# 3 to_device
b = {"x": torch.arange(3.0), "radar": {"a": torch.ones(2)}, "steps": [[(torch.zeros(1), torch.ones(1))]], "tokens": ["t"]}
o1, o2 = T.to_device(b, "cpu"), T.to_device(b, "cpu", non_blocking=True)
check(torch.equal(o1["x"], o2["x"]) and torch.equal(o1["radar"]["a"], o2["radar"]["a"]) and o2["tokens"] == ["t"] and torch.equal(o2["steps"][0][0][1], torch.ones(1)), "to_device(non_blocking=True) gives the same batch")

# 4 the resume
c = T.TrainConfig()
check((c.pin_memory, c.prefetch_factor, c.cudnn_benchmark, c.tf32, c.profile_steps) == (False, 0, False, False, 0), "all new fields are off by default")
check({"pin_memory", "prefetch_factor", "profile_steps"} <= set(T.RESUME_FREE) and not ({"cudnn_benchmark", "tf32"} & set(T.RESUME_FREE)), "speed-only fields may change on a resume, the ones that change numbers may not")

# 5 the profiler
d = tempfile.mkdtemp()
pr = probe.StepProfiler(d, steps=3, warmup=2, device="cpu")
w = torch.randn(64, 64)
for i in range(9):
    (w @ w).sum().item()
    pr.tick()
txt = open(os.path.join(d, "profile.txt")).read()
check(pr.done and "window: 3 optimiser steps" in txt and "no CUDA in this run" in txt and "aten::item" in txt and "top operators by CPU time:" in txt, "profile.txt after warm-up + window: the window, the sync calls, the operators")
mt = os.path.getmtime(os.path.join(d, "profile.txt"))
pr.tick()
pr.close()
check(os.path.getmtime(os.path.join(d, "profile.txt")) == mt and not os.path.exists(os.path.join(d, "profile.txt.tmp")), "written once, atomically")
d2 = tempfile.mkdtemp()
pe = probe.StepProfiler(d2, steps=50, warmup=1, device="cpu")
for i in range(4):
    (w @ w).sum().item()
    pe.tick()
pe.close()
check("window: 3 optimiser steps" in open(os.path.join(d2, "profile.txt")).read(), "a run that ends inside the window writes what it has")
d3 = tempfile.mkdtemp()
po = probe.StepProfiler(d3, steps=0, warmup=1, device="cpu")
for i in range(5):
    po.tick()
po.close()
check(not os.path.exists(os.path.join(d3, "profile.txt")), "steps = 0: off, nothing written")
d4 = tempfile.mkdtemp()
pn = probe.StepProfiler(d4, steps=5, warmup=100, device="cpu")
for i in range(3):
    pn.tick()
pn.close()
check(not os.path.exists(os.path.join(d4, "profile.txt")), "a run that ends before the warm-up has nothing to write")

# 6 the resume check knows the two switches that change numbers (a checkpoint from before them has no such key)
old_cfg = {k: v for k, v in dataclasses.asdict(T.TrainConfig()).items() if k not in ("cudnn_benchmark", "tf32", "pin_memory", "prefetch_factor", "profile_steps")}


def refused(**kw):
    try:
        T.check_resume(old_cfg, T.TrainConfig(**kw), "toy")
        return False
    except SystemExit:
        return True


check(refused(tf32=True) and refused(cudnn_benchmark=True) and not refused() and not refused(pin_memory=True, prefetch_factor=4, profile_steps=5),
      "resume of a checkpoint from before the switches: tf32 / cudnn_benchmark refused, the speed-only fields and the defaults accepted")

# 7 the profiler: a window of at most MAX_STEPS, a failure is printed and never raised, one line per sync kind
d5 = tempfile.mkdtemp()
pw = probe.StepProfiler(d5, steps=30, warmup=1, device="cpu")
for i in range(40):
    (w @ w).sum().item()
    pw.tick()
txt5 = open(os.path.join(d5, "profile.txt")).read()
check(pw.steps == probe.StepProfiler.MAX_STEPS == 8 and "window: 8 optimiser steps" in txt5, "a window of 30 steps is shortened to 8 (30 steps held 22 GB of host RAM)")
check("_local_scalar_dense" not in txt5.split("\n")[2], "the sync line names aten::item once, not also its implementation")
d6 = tempfile.mkdtemp()
pf = probe.StepProfiler(d6, steps=2, warmup=1, device="cpu")
pf._write = lambda: (_ for _ in ()).throw(RuntimeError("report failed"))
for i in range(4):
    pf.tick()
check(pf.done and pf.prof is None and not os.path.exists(os.path.join(d6, "profile.txt")), "a failing report is printed, not raised, and the profiler is freed")

# 8 the REAL run_clip against the one it replaced (the old text is kept here as the reference), with a fake forward: same terms, order, totals, gradients and Kendall values
OLD_RUN_CLIP = '''
def run_clip(model, loss, clip, step_frac, device, amp, cfg, diag=False, train_terms=True, backward=False, augment=False):
    L, state, prev, acc, totals, dg = len(clip), None, None, {}, [], None
    bptt, accum = bool(cfg is not None and cfg.seq_bptt), (cfg.accum if cfg is not None else 1)
    for t, batch in enumerate(clip):
        step = clip_step(prev, device) if prev is not None else None
        out, res, _ = forward_loss(model, loss, batch, step_frac, device, amp, cfg, diag=diag and t == L - 1,
                                   train_terms=train_terms, state=state, step=step)
        if not torch.isfinite(out["total"]):
            return acc, False, dg, totals
        totals.append(_f(out["total"]))
        if backward:
            (out["total"] / (L * accum)).backward(retain_graph=bptt and t < L - 1)
        for k, v in out.items():
            if k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in LOGGED_TERMS:
                acc[k] = acc.get(k, 0.0) + _f(v) / L
        if "diag" in out:
            dg = out["diag"]
        state = res["state"] if bptt else detach_tree(res["state"])
        prev = batch
    return acc, True, dg, totals
'''
ns = dict(vars(T))
exec(OLD_RUN_CLIP, ns)                                                           # the HEAD text of run_clip (augment and the docstring left out), in train's namespace
logged = sorted(T.LOGGED_TERMS)[:4]


def fake_forward_factory(w, kendall, nan_at):
    calls = [0]

    def fake_forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
        i = calls[0]
        calls[0] += 1
        with torch.no_grad():
            kendall.clamp_(-0.25, 0.25)                                         # an in-place change of a Parameter, like TaskWeights
        tot = (w * batch["x"]).sum()
        if nan_at is not None and i == nan_at:
            tot = tot * float("nan")
        out = {"total": tot, "group.areas": (w ** 2).sum(), "group.lines": w.sum() * 0.5, "s.areas": kendall[0], "s.lines": 1.0, "consistency": 0.25 * i, "not_logged": torch.tensor(5.0),
               logged[0]: (w[0] * 2).bfloat16(), logged[1]: 0.5 + i, logged[2]: torch.tensor(i / 3 + 2.0 ** -40, dtype=torch.float64), logged[3]: w[1:2] * 3, "diag": {"a": 1}}
        with torch.no_grad():
            kendall.mul_(0.9)                                                   # changed again after the read
        return out, {"state": {"m": w.detach() * 2}}, None
    return fake_forward_loss


combos = same_all = 0
for nan_at in (None, 0, 1, 2):
    for bptt in (False, True):
        for L_ in (1, 3):
            res = {}
            for name, fn, glob in (("old", ns["run_clip"], ns), ("new", T.run_clip, vars(T))):
                torch.manual_seed(0)
                w, kendall = nn.Parameter(torch.tensor([0.7, -0.4, 1.3])), nn.Parameter(torch.tensor([0.3, -0.2]))
                keep = glob["forward_loss"], glob["clip_step"]
                glob["forward_loss"], glob["clip_step"] = fake_forward_factory(w, kendall, nan_at), (lambda prev, device: (0.5, 1.0, 0.0))
                try:
                    cfg_ = T.TrainConfig(accum=2, seq_bptt=bptt, flip_aug=0.0)
                    clip = [{"x": torch.tensor([1.0 + t, 2.0, -1.0]), "tokens": ["a"]} for t in range(L_)]
                    r = fn(None, None, clip, 0.3, "cpu", "off", cfg_, diag=False, backward=True, augment=False)
                finally:
                    glob["forward_loss"], glob["clip_step"] = keep
                res[name] = (r, None if w.grad is None else w.grad.clone(), kendall.detach().clone())
            (ro, go, ko), (rn, gn, kn) = res["old"], res["new"]
            ok_ = (list(ro[0]) == list(rn[0]) and ro[0] == rn[0] and ro[1] == rn[1] and ro[2] == rn[2] and ro[3] == rn[3] and (go is None) == (gn is None)
                   and (go is None or torch.equal(go, gn)) and torch.equal(ko, kn))
            combos += 1
            same_all += bool(ok_)
check(combos == 16 and same_all == 16 and len(ro[0]) > 5, f"run_clip: the same terms (and order), finite flag, totals, gradients and Kendall values as the per-term float() version in {same_all} of {combos} cases (NaN at keyframe 0/1/2, bptt, 1 and 3 keyframes, bf16 and float64 terms)")

print("SPEED FLAGS GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)

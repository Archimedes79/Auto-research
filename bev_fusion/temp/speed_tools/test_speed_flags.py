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

print("SPEED FLAGS GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)

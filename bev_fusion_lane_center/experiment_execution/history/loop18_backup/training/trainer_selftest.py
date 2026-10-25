"""Self-test of trainer.fit: (1) a run that crashes at step 120 and is restarted ends with the same weights as an uninterrupted run, (2) no .done flag after the crash, (3) a non-finite loss is skipped,
(4) the guard rolls back after a loss spike.  Toy regression on the CPU, temporary run folders.   Run: tools/env/_py14.sh trainer_selftest.py"""
import os
import shutil
import tempfile

import torch

import trainer as T

tmp = tempfile.mkdtemp()


def make_model():
    torch.manual_seed(3)
    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))


W = torch.randn(8, 1, generator=torch.Generator().manual_seed(7))


def batch_fn(step):
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randn(64, 8, generator=g)
    return x, x @ W


def loss_fn(model, b):
    x, y = b
    l = torch.nn.functional.mse_loss(model(x), y)
    return l, {"mse": float(l)}


kw = dict(steps=200, batch_fn=batch_fn, lr=3e-3, amp="none", device="cpu", ckpt_every=50, log_every=50, clip=1.0, guard=False)
# (1) uninterrupted
ma = make_model()
T.fit(ma, loss_fn, T.Run("a", tmp), **kw)


# (1b) crash at step 120, then resume
class Boom(Exception):
    pass


calls = {"n": 0}


def crashy(model, b):
    calls["n"] += 1
    if calls["n"] == 121:
        raise Boom("simulated crash")
    return loss_fn(model, b)


mb = make_model()
rb = T.Run("b", tmp)
try:
    T.fit(mb, crashy, rb, **kw)
    raise SystemExit("expected a crash")
except Boom:
    pass
assert not os.path.exists(rb.path(".done")), "done flag after a crash"
assert os.path.exists(rb.path("ckpt.pt")) and rb._status["state"] == "failed"
mc = make_model()
T.fit(mc, loss_fn, T.Run("b", tmp), **kw)  # same run name: resumes from the checkpoint of step 100
same = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(ma.parameters(), mc.parameters()))
print("resumed run equals the uninterrupted run:", same, "| .done flag:", os.path.exists(os.path.join(tmp, "b", ".done")))
assert same and os.path.exists(os.path.join(tmp, "b", ".done"))
# (3) non-finite loss
n = {"i": 0}


def nan_once(model, b):
    n["i"] += 1
    l, info = loss_fn(model, b)
    return (l * float("nan") if n["i"] == 10 else l), info


T.fit(make_model(), nan_once, T.Run("c", tmp), **{**kw, "steps": 40, "ckpt_every": 20})
st = T.Run("c", tmp)._status
print("non-finite step skipped:", st["skipped"] == 1)
assert st["skipped"] == 1
# (4) guard: a spike in the loss after step 100 forces a rollback
n2 = {"i": 0}


def spike(model, b):
    n2["i"] += 1
    l, info = loss_fn(model, b)
    return (l * 1e4 if 101 <= n2["i"] <= 160 else l), info


T.fit(make_model(), spike, T.Run("d", tmp), **{**kw, "steps": 300, "guard": True, "log_every": 20, "ckpt_every": 20, "guard_ratio": 1.5})
st = T.Run("d", tmp)._status
print("rollbacks after a loss spike:", st["rollbacks"])
assert st["rollbacks"] >= 1
shutil.rmtree(tmp, ignore_errors=True)
print("trainer selftest passed")

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


# (5) a FINITE loss with a non-finite gradient (sqrt at 0) is skipped like a non-finite loss: no NaN weight, the run still finishes
def sqrt_at_zero(model, b):
    l, info = loss_fn(model, b)
    w = model[0].weight
    return l + torch.sqrt((w * 0).pow(2).sum()), info                    # value 0, gradient nan


n3 = {"i": 0}


def nan_grad_once(model, b):
    n3["i"] += 1
    return sqrt_at_zero(model, b) if n3["i"] == 10 else loss_fn(model, b)


mg = make_model()
T.fit(mg, nan_grad_once, T.Run("e", tmp), **{**kw, "steps": 40, "ckpt_every": 20})
st = T.Run("e", tmp)._status
assert st["state"] == "done" and st["skipped"] == 1 and st["updates"] == 39, st
assert all(torch.isfinite(p).all() for p in mg.parameters()), "a non-finite gradient must never reach the weights"
print("finite loss with a non-finite gradient skipped, weights finite:", True)


# (6) a run that never updates a weight, or whose steps are non-finite in a row, FAILS (no .done flag, state failed)
def must_fail(name, fn, what, **over):
    r = T.Run(name, tmp)
    try:
        T.fit(make_model(), fn, r, **{**kw, "steps": 40, "ckpt_every": 20, **over})
    except RuntimeError as e:
        assert what in str(e), str(e)
    else:
        raise AssertionError(f"{name}: expected the run to fail")
    assert not os.path.exists(r.path(".done")) and not os.path.exists(r.path("model.pt")) and T.Run(name, tmp)._status["state"] == "failed"
    return T.Run(name, tmp)._status


st = must_fail("f", lambda m, b: (loss_fn(m, b)[0] * float("nan"), {}), "in a row", max_bad=5)             # every loss nan: stops after 5
assert st["step"] == 5, st
st = must_fail("g", lambda m, b: sqrt_at_zero(m, b), "in a row", max_bad=5)                                  # every gradient nan
assert st["step"] == 5, st
must_fail("h", lambda m, b: (loss_fn(m, b)[0] * float("nan"), {}), "no optimizer update", steps=10, max_bad=20)   # all 10 skipped, fewer than max_bad
print("a diverged or all-skipped run fails: no .done flag, no model.pt, status failed")



# (6b) "in a row": two bursts of max_bad - 1 non-finite steps with good steps between them finish; the count restarts after a good step
n4 = {"i": 0}


def bursts(model, b):
    n4["i"] += 1
    l, info = loss_fn(model, b)
    return (l * float("nan") if n4["i"] in (3, 4, 5, 6, 20, 21, 22, 23) else l), info


T.fit(make_model(), bursts, T.Run("j", tmp), **{**kw, "steps": 60, "ckpt_every": 20, "max_bad": 5})
st = T.Run("j", tmp)._status
assert st["state"] == "done" and st["skipped"] == 8 and st["updates"] == 52, st
assert abs(st["skipped_share"] - 8 / 60) < 1e-3 and "WARNING" in open(os.path.join(tmp, "j", "log.txt")).read(), st      # 13 % skipped: finished, with a warning
assert T.Run("e", tmp)._status["skipped_share"] == 0.025 and "WARNING" not in open(os.path.join(tmp, "e", "log.txt")).read()   # 1 of 40: no warning
# a skipped step that lands on a checkpoint multiple still writes the checkpoint (and a crash right after resumes from it)
n5 = {"i": 0}


def nan_on_20_then_crash(model, b):
    n5["i"] += 1
    if n5["i"] == 25:
        raise RuntimeError("simulated crash")
    l, info = loss_fn(model, b)
    return (l * float("nan") if n5["i"] == 20 else l), info


try:
    T.fit(make_model(), nan_on_20_then_crash, T.Run("k", tmp), **{**kw, "steps": 40, "ckpt_every": 20})
    raise AssertionError("expected the simulated crash")
except RuntimeError:
    pass
assert torch.load(os.path.join(tmp, "k", "ckpt.pt"), weights_only=False)["step"] == 20, "a skipped step must not skip the checkpoint"


# (6c) the fp16 scaler skips an update itself (it lowers its scale): counted like a non-finite step, not as an update. A stand-in scaler on the CPU.
class FakeScaler:
    inf_calls = set()

    def __init__(self, enabled=True):
        self.n, self._s, self.skip = 0, 1024.0, False

    def is_enabled(self):
        return True

    def get_scale(self):
        return self._s

    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        self.n += 1
        self.skip = self.n in FakeScaler.inf_calls
        if not self.skip:
            opt.step()

    def update(self):
        if self.skip:
            self._s /= 2

    def state_dict(self):
        return {}

    def load_state_dict(self, s):
        pass


real_scaler = T.torch.cuda.amp.GradScaler
T.torch.cuda.amp.GradScaler = FakeScaler
try:
    FakeScaler.inf_calls = {10, 11, 12}
    T.fit(make_model(), loss_fn, T.Run("l", tmp), **{**kw, "steps": 40, "ckpt_every": 20, "amp": "fp16"})
    st = T.Run("l", tmp)._status
    assert st["state"] == "done" and st["skipped"] == 3 and st["updates"] == 37, st
    FakeScaler.inf_calls = set(range(1, 100))
    st = must_fail("m", loss_fn, "in a row", amp="fp16", max_bad=6)
    assert st["step"] == 6, st
finally:
    T.torch.cuda.amp.GradScaler = real_scaler
print("bursts shorter than max_bad finish; a skipped step keeps the checkpoint cadence; the fp16 scaler's skips are counted")


# (7) a data loader is iterated anew on every pass (itertools.cycle replays the first pass); an empty loader is an error
class Loader:
    def __init__(self):
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter([self.passes * 10 + 1, self.passes * 10 + 2])


ld = Loader()
e = T._endless(ld)
assert [next(e) for _ in range(6)] == [11, 12, 21, 22, 31, 32] and ld.passes == 3
try:
    next(T._endless([]))
    raise AssertionError("an empty loader must raise")
except ValueError:
    pass
ds = torch.utils.data.TensorDataset(*batch_fn(0))
seen = []
T.fit(make_model(), lambda m, b: (seen.append(float(b[0].sum())) or loss_fn(m, b)), T.Run("i", tmp),
      **{**kw, "steps": 6, "batch_fn": None, "loader": torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True), "ckpt_every": 0})
assert len(seen) == 6 and seen[0:2] != seen[2:4] and seen[2:4] != seen[4:6], seen                      # 2 batches per epoch, reshuffled each epoch
print("loader: a new pass every epoch, reshuffled:", True)
shutil.rmtree(tmp, ignore_errors=True)
print("trainer selftest passed")

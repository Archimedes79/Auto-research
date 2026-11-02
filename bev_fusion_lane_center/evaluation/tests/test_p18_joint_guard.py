"""Spec test of the update guard of training/train/p18_joint.py (CPU, a stand-in for the fp16 GradScaler as in trainer_selftest.py): a finite loss with inf / nan
gradients makes the fp16 scaler skip the update and lower its scale; that is a bad step, not an update, and 20 in a row fail the run (the joint trainer once counted only
non-finite losses, so a run whose updates were all skipped finished with an unchanged builder).
Run (from the problem folder): wsl bash tools/env/_py14.sh test_p18_joint_guard.py"""
import torch

import p18_joint as J


class FakeScaler:
    """skips the optimizer step and halves its scale on the calls in `inf_calls` (numbered from 1), like GradScaler on an inf gradient"""

    def __init__(self, inf_calls, grow_every=0):
        self.n, self.inf_calls, self._s, self.skip, self.grow_every, self.ok_run = 0, inf_calls, 1024.0, False, grow_every, 0

    def get_scale(self):
        return self._s

    def step(self, opt):
        self.n += 1
        self.skip = self.n in self.inf_calls
        if not self.skip:
            opt.step()

    def update(self):
        if self.skip:
            self._s /= 2
            self.ok_run = 0
        else:
            self.ok_run += 1
            if self.grow_every and self.ok_run % self.grow_every == 0:       # GradScaler: the scale grows after growth_interval steps without an inf
                self._s *= 2


def run_steps(inf_calls, n, grow_every=0):
    """n guarded steps on a one-weight model; (weight, updates made, bad count at the end)"""
    w = torch.nn.Parameter(torch.zeros(1))
    opt, sc, bad, ups = torch.optim.SGD([w], lr=1.0), FakeScaler(inf_calls, grow_every), 0, 0
    for i in range(n):
        w.grad = torch.ones(1)
        bad, updated = J.guarded_step(sc, opt, bad, i)
        ups += updated
    return float(w), ups, bad


def test_scale_drop_is_a_bad_step_not_an_update():
    w, ups, bad = run_steps({3, 4}, 10)
    assert ups == 8 and w == -8.0 and bad == 0, (w, ups, bad)  # two skipped steps: no update, the count resets on the next update


def test_a_growing_scale_is_an_update():
    """the scale grows every growth_interval good steps: that step DID update (a `scale == scale0` rule would count it as skipped)"""
    w, ups, bad = run_steps(set(), 10, grow_every=3)
    assert ups == 10 and w == -10.0 and bad == 0, (w, ups, bad)
    w, ups, bad = run_steps({5}, 10, grow_every=3)
    assert ups == 9 and w == -9.0 and bad == 0, (w, ups, bad)


def test_main_goes_through_the_guard():
    """the loop of main() takes its optimizer step only through guarded_step and does not count a skipped step as an iteration"""
    import inspect
    src = inspect.getsource(J.main)
    assert "guarded_step(scaler, opt, bad_steps, it)" in src and "scaler.step(" not in src, "the optimizer step must go through guarded_step"
    after = src.split("guarded_step(scaler, opt, bad_steps, it)")[1]
    assert "continue" in after and after.index("continue") < after.index("sched.step()") < after.index("it += 1"), "a skipped step must `continue` before sched.step() and the iteration count"


def test_a_run_whose_updates_are_all_skipped_fails():
    try:
        run_steps(set(range(1, 100)), 100)
    except RuntimeError as e:
        assert "20 steps without an update in a row" in str(e), e
        return
    raise AssertionError("a run in which every update was skipped did not fail")


def test_loss_and_gradient_skips_share_one_count():
    bad = 0
    for i in range(J.MAX_BAD - 1):
        bad = J.count_bad(bad, False, "non-finite loss", i)
    assert bad == J.MAX_BAD - 1 and J.count_bad(bad, True, "", 0) == 0
    try:
        J.count_bad(bad, False, "non-finite gradient", 99)
    except RuntimeError:
        return
    raise AssertionError("the 20th bad step in a row did not raise")


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_p18_joint_guard: {n} tests passed")

"""scratch: does the trainkit selftest fail on the mutants it was extended for? (run from autoresearch/ with the gcar python)"""
import contextlib
import io
import sys

sys.path.insert(0, "../bev_fusion/tests")
sys.path.insert(0, ".")
import trainkit.checkpoint as ck  # noqa: E402
import trainkit.ema as em  # noqa: E402
import trainkit.loop as lp  # noqa: E402
import trainkit.selftest as ST  # noqa: E402
import torch  # noqa: E402

src = lambda m: open(m.__file__, encoding="utf-8").read()  # noqa: E731


def mutated(module, old, new, count=1):
    import types
    s = src(module)
    assert s.count(old) == count, old
    c = types.ModuleType(module.__name__ + "_m")
    c.__package__, c.__file__ = module.__package__, module.__file__
    exec(compile(s.replace(old, new), module.__file__, "exec"), c.__dict__)
    return c


tmp_prune = 'if f.startswith("ckpt_") and f.endswith(".pt.tmp"):\n            os.remove(os.path.join(out_dir, f))'
cases = [
    ("step_frac 0", ST, "fit", mutated(lp, "StepInfo(step, step / max(1, cfg.steps), m, cfg.accum, scaler)", "StepInfo(step, 0.0, m, cfg.accum, scaler)").fit),
    ("no prune after the stop save", ST, "fit", mutated(lp, "            save(step)\n            checkpoint.prune_checkpoints(out_dir, cfg.keep)\n            control.mark_stopped", "            save(step)\n            control.mark_stopped").fit),
    ("prune leaves .tmp", ck, "prune_checkpoints", mutated(ck, tmp_prune, tmp_prune.replace("os.remove(os.path.join(out_dir, f))", "pass")).prune_checkpoints),
    ("torch RNG not restored", torch, "set_rng_state", lambda s: None),
    ("EMA without the cap", em.EMA, "update", mutated(em, "d = min(self.decay, (1 + step) / (10 + step))", "d = self.decay").EMA.update),
]
for label, obj, attr, new in cases:
    orig = getattr(obj, attr)
    setattr(obj, attr, new)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ST.main()
        print("SURVIVED:", label)
    except AssertionError as e:
        print("killed:  ", label, "|", str(e)[:90])
    finally:
        setattr(obj, attr, orig)

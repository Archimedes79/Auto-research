"""scratch: tests/test_velocity_hurdle.py must FAIL when the loop warm-starts also on a resume (`if cfg.init_from:` without 'no checkpoint yet')"""
import runpy
import sys
import traceback

sys.path.insert(0, "tests")
sys.path.insert(0, "training")
import test_trainkit_equivalence as E  # noqa: E402
import trainkit.loop as lp  # noqa: E402

if "--mutant" in sys.argv:
    E.NEW.fit = E.mutated(lp, "if resumed is None and cfg.init_from:", "if cfg.init_from:").fit
try:
    runpy.run_path("tests/test_velocity_hurdle.py", run_name="__main__")
    print("HURDLE TEST PASSED", "(mutant on)" if "--mutant" in sys.argv else "(unmutated)")
except BaseException as e:  # noqa: BLE001
    print("HURDLE TEST FAILED:", type(e).__name__, str(e)[:160], "(mutant on)" if "--mutant" in sys.argv else "(unmutated)")

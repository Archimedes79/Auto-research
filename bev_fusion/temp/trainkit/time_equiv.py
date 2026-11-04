"""scratch: where does the time of tests/test_trainkit_equivalence.py go"""
import cProfile
import pstats
import sys
import tempfile
import time

sys.path.insert(0, "tests")
import test_trainkit_equivalence as E  # noqa: E402

t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    OLD = E.load_old(tmp)
    t1 = time.time()
    E.scenarios(OLD)
    t2 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    E.kill_mutants(OLD)
    pr.disable()
    t3 = time.time()
print(f"import+load {t1 - t0:.1f}s  scenarios {t2 - t1:.1f}s  mutants {t3 - t2:.1f}s")
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)

"""The tests of the evaluation: known answers of the engine, the class interface and the command line (tests/) and of the comparison core (selftest_comparison.py).

    python -m Evaluation.selftest [word ...]          (from autoresearch/; needs numpy and matplotlib, the torch checks are skipped without torch)
The words filter the checks by name (`comparison`: what comparison/mutation_check.py runs). Exit 0 all pass, 1 a check fails (its traceback is printed), 2 no check has such a name."""
from __future__ import annotations

import sys
import time
import traceback

from . import selftest_comparison, tests
from .tests import test_api, test_cli, test_engine                  # noqa: F401  (each registers its checks in tests.CHECKS)


def main(only=()) -> int:
    checks = tests.CHECKS + selftest_comparison.CHECKS
    if only:
        checks = [c for c in checks if any(o in c[0] for o in only)]
        if not checks:
            print(f"Evaluation selftest: no check matches {list(only)}")
            return 2
    t0, bad = time.time(), 0
    for name, fn in checks:
        t = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f} s)", flush=True)
        except Exception:                                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
    print(f"Evaluation selftest: {len(checks) - bad}/{len(checks)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

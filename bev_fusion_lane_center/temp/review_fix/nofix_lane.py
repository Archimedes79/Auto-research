"""Throw-away: the dp_eps tests of test_eval_parts against a route_asm that keeps dp_eps (both must fail)."""
import os
import sys

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, f"{R}/evaluation/tests")
import eval_parts as EP  # noqa: E402
import test_eval_parts as T  # noqa: E402

EP.route_asm = lambda e: e["asm"]
for name in ("test_route_builder_drops_dp_eps_and_keeps_the_rest_of_the_assembly", "test_route_series_drops_dp_eps_too"):
    try:
        getattr(T, name)()
        print("NOFIX:", name, "PASSED without the fix (worthless)")
    except AssertionError as e:
        print("NOFIX:", name, "fails without the fix, as it must:", str(e)[:100])

import importlib.util, sys
sys.path.insert(0, ".")
import evaluation.bev_timing as new
spec = importlib.util.spec_from_file_location("old_timing", "temp/pitch/bev_timing_before.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
new.busy_processes = old.busy_processes          # the selftest calls the module-level name: the OLD logic under the NEW test
try:
    new.selftest()
    print("UNEXPECTED: the new test passes with the old logic")
except AssertionError as e:
    print("the new test FAILS with the old logic, as it must:", str(e)[:140])

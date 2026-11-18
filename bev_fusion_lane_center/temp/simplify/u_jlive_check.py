"""U3-U5 scratch: test_jlive_train's full run (selftest, real-data check, the hand mutants) on the EDITED jlive_train.py (the test's own modes load the live file by path; its mutant children get no import finder):
the test source of the tree, the mutants applied to the edited file, each child started through ux_run_with.py (so it imports the edited modules too). Run through ux_run_with.py with U2_WT set."""
import os
import subprocess
import sys
import tempfile

import jlive_train                                                                                   # through the finder of ux_run_with: the edited file

ROOT = os.environ["BFLC_ROOT"]
path = jlive_train.__file__
src_path = os.path.join(os.path.dirname(os.path.dirname(path)), "grid", "test_jlive_train.py")
ns = {"__name__": "test_jlive_train_u", "__file__": os.path.join(ROOT, "training", "grid", "test_jlive_train.py")}
exec(compile(open(src_path, encoding="utf-8").read(), ns["__file__"], "exec"), ns)
print(f"jlive check: selftest, real_check and the mutants on {path} (test source {src_path})", flush=True)
ns["selftest"](path)
ns["real_check"](path)
import test_lane_live as TLL  # noqa: E402

runner = os.path.join(ROOT, "temp", "simplify", "ux_run_with.py")
src, left = open(path, encoding="utf-8").read(), []
with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
    for k, (name, a, b) in enumerate(ns["MUTANTS"]):
        if src.count(a) != 1:
            res = f"NOT APPLIED ({src.count(a)} matches)"
        else:
            p = os.path.join(tmp, f"m{k}.py")
            open(p, "w", encoding="utf-8").write(src.replace(a, b))
            r = subprocess.run([sys.executable, "-c", TLL.LAUNCH, runner, "training/grid/test_jlive_train.py", "one", p], capture_output=True, text=True, env=dict(ns["ENV"], U2_WT=os.environ["U2_WT"]))
            lines = [ln for ln in (r.stderr or r.stdout).strip().splitlines() if not ln.startswith("u2_run_with")] or ["no output"]
            tail = lines[-1][:150]
            res = "SURVIVED" if r.returncode == 0 else ("killed" if tail.startswith(("AssertionError", "Kill", "jlive_train_under_test.Kill")) else "CRASHED: " + tail)
        print(f"{res[:170]:<12} {name}", flush=True)
        if not res.startswith("killed"):
            left.append(name)
print(f"{len(ns['MUTANTS']) - len(left)} of {len(ns['MUTANTS'])} killed", flush=True)
assert not left, f"mutants survived or crashed: {left}"
print("jlive check ok", flush=True)

"""throw-away: run a test file with the scratch copies jtoken_train_new.py / jtoken_eval_new.py loaded in place of jtoken_train / jtoken_eval (before they are moved in).
    wsl bash tools/env/_py14.sh temp/e372fix/run_new_e372fix.py <test file relative to the problem folder> [test names]"""
import importlib.util
import os
import runpy
import sys

ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
for mod, f in (("jtoken_train", os.environ.get("TRAIN_FILE", "jtoken_train_new.py")), ("jtoken_eval", "jtoken_eval_new.py")):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    print(f"{mod} <- {m.__file__}", flush=True)
test = os.path.join(ROOT, sys.argv[1])
sys.argv = [test] + sys.argv[2:]
runpy.run_path(test, run_name="__main__")

"""scratch: run the scratch copy of test_jtoken_eval.py against the scratch copy of jtoken_eval.py (both in this folder) before they are moved into evaluation/.
usage: bash tools/env/_py14.sh temp/e377/e377a_session_scratch_/e377a_runtests.py [test names]"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["BFLC_ROOT"]
for sub_ in ("..", "training", "training/grid", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
sys.path.insert(0, HERE)
import jtoken_eval  # noqa: E402

assert os.path.dirname(os.path.abspath(jtoken_eval.__file__)) == HERE, jtoken_eval.__file__
sys.argv = [os.path.join(HERE, "test_jtoken_eval.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

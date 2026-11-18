"""U1 scratch: the scratch test copy run against the scratch module copy (both are moved into evaluation/ with mv when green); args = test names"""
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL_TEST = os.path.join(ROOT, "evaluation", "tests", "test_jtoken_eval.py")
spec = importlib.util.spec_from_file_location("jtoken_eval", os.path.join(HERE, "u1_jtoken_eval.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["jtoken_eval"] = mod
spec.loader.exec_module(mod)
src = open(os.path.join(HERE, "u1_test_jtoken_eval.py"), encoding="utf-8").read()
exec(compile(src, REAL_TEST, "exec"), {"__name__": "__main__", "__file__": REAL_TEST})

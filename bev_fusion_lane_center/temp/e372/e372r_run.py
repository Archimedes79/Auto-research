"""throw-away: run the scratch copy of the jtoken_eval tests against the scratch copy of jtoken_eval.py (loaded as `jtoken_eval`), so that the live files stay untouched while other chats run them"""
import importlib.util
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("jtoken_eval", os.path.join(HERE, "e372r_jtoken_eval.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["jtoken_eval"] = mod
spec.loader.exec_module(mod)
test = os.path.join(HERE, "e372r_test_jtoken_eval.py")
sys.argv = [test] + sys.argv[1:]
runpy.run_path(test, run_name="__main__")

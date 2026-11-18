"""U1 scratch: run a project script (path relative to the problem folder) with the edited module temp/simplify/u1_jtoken_eval.py preloaded as `jtoken_eval`
(sys.modules wins over every sys.path insert the script makes); cwd and sys.path[0] = the script's folder, as `_py14.sh` sets them. A subprocess of the script does NOT get it.
usage: _py14.sh u1_run_with.py <script relative to the problem folder> [args]"""
import importlib.util
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("jtoken_eval", os.path.join(HERE, "u1_jtoken_eval.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["jtoken_eval"] = mod
spec.loader.exec_module(mod)
script = os.path.join(ROOT, sys.argv[1])
sys.argv = [script] + sys.argv[2:]
os.chdir(os.path.dirname(script))
sys.path.insert(0, os.path.dirname(script))
print(f"u1_run_with: {sys.argv[1:]} on {script} with jtoken_eval = {mod.__file__}", flush=True)
runpy.run_path(script, run_name="__main__")
assert sys.modules["jtoken_eval"] is mod, "the script replaced the preloaded module"

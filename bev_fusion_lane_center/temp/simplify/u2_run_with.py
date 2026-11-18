"""U2 scratch: run a script of the worktree C:/Auto-research_wt_simplify (path relative to the problem folder) against the LIVE data: an import finder put first on sys.meta_path loads the edited modules
of the worktree under their names WHEN they are imported (a finder wins over every sys.path insert the script makes; nothing is imported early), the script's own source is the worktree's (a scratch script under
temp/: the live one), compiled with the LIVE path as __file__ (so its paths and sys.path inserts are the live tree's); cwd and sys.path[0] = the live folder of the script, as `_py14.sh` sets them. A subprocess
of the script does NOT get the finder.
usage: _py14.sh u2_run_with.py <script relative to the problem folder> [args]"""
import importlib.abc
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WT = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Auto-research_wt_simplify", os.path.basename(ROOT))
MODULES = {"jtoken": "training/grid/jtoken.py", "jtoken_train": "training/train/jtoken_train.py", "jlive_train": "training/train/jlive_train.py", "jtoken_eval": "evaluation/jtoken_eval.py"}
MODULES.update(dict(m.split("=", 1) for m in os.environ.get("U2_EXTRA", "").split(";") if m))                       # name=path (absolute) of more replaced modules


class Worktree(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        return importlib.util.spec_from_file_location(name, os.path.join(WT, MODULES[name])) if name in MODULES else None


sys.meta_path.insert(0, Worktree())
script = os.path.join(ROOT, sys.argv[1])
src_path = os.environ.get("U2_SRC") or (os.path.join(WT, sys.argv[1]) if os.path.exists(os.path.join(WT, sys.argv[1])) else script)    # U2_SRC: another source; a scratch script (temp/ is not in git): the live one
src = open(src_path, encoding="utf-8").read()
sys.argv = [script] + sys.argv[2:]
os.chdir(os.path.dirname(script))
sys.path.insert(0, os.path.dirname(script))
print(f"u2_run_with: {sys.argv[1:]} on {script} (source {src_path}); {', '.join(MODULES)} from {WT}", flush=True)
try:
    exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})
finally:
    print("u2_run_with: imported " + ", ".join(f"{n} = {sys.modules[n].__file__}" for n in MODULES if n in sys.modules), flush=True)
    assert all(sys.modules[n].__file__ == os.path.join(WT, MODULES[n]) for n in MODULES if n in sys.modules), "a replaced module came from elsewhere"

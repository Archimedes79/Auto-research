"""scratch: does the evaluation adapter load a run from its code snapshot? usage: check_eval_load.py <run folder>   (one run per process)"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import bev_adapter as A  # noqa: E402

run = sys.argv[1]
A.setup_paths(run)
cfg, model, loss, step, path = A.load_run(run, "cpu")
import train as T  # noqa: E402
print("run", run, "step", step, "ckpt", os.path.basename(path), "| train.py from", os.path.relpath(T.__file__, ROOT),
      "| trainkit from", os.path.relpath(T.tk_ckpt.__file__, ROOT) if hasattr(T, "tk_ckpt") else "- (self-contained trainer of an old run)")

"""E311 refinements: the snapped read-out with snap_all / snap_guard / snap_dedup, E305 / E287 p2 dumps, through the scorer of record; writes temp/gt2/snapopt/<run>_<opts>_thr<t>.json. usage: snapopt_sweep.py <run> <seed> <thr> <opts>   opts: plain|all|guard|dedup|guard+dedup|all+guard+dedup"""
import builtins
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qends_eval as QE

run, seed, thr, opts = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
flags = {} if opts == "plain" else {f"snap_{k}": True for k in opts.split("+")}
os.makedirs(os.path.join(ROOT, "temp", "gt2", "snapopt"), exist_ok=True)
target = os.path.join(ROOT, "temp", "gt2", "snapopt", f"{run}_{opts}_thr{thr}.json")


def safe_open(path, mode="r", *a, **k):
    if "ends_card" in str(path) and "w" in mode:
        return builtins.open(target, mode, *a, **k)
    return builtins.open(path, mode, *a, **k)


QE.open = safe_open
QE.cmd_score(SimpleNamespace(run=run, seed=seed, thr=thr, full=None, base=None, skip_base_check=True, chord=False, snap=True, **flags))
print("SNAPOPT", opts, "done", flush=True)

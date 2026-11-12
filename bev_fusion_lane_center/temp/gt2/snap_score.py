"""the delivered graph with SNAPPED connectors (temp/jr/snapconn.py) scored by the scorer of record: usage snap_score.py <run> <seed> <thr> [<thr> ...]; writes temp/gt2/snap/<run>_thr<t>.json, the run's cards are not touched"""
import builtins
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qends_eval as QE

run, seed, thrs = sys.argv[1], int(sys.argv[2]), [float(x) for x in sys.argv[3:]]
os.makedirs(os.path.join(ROOT, "temp", "gt2", "snap"), exist_ok=True)
state = {}


def safe_open(path, mode="r", *a, **k):
    if "ends_card" in str(path) and "w" in mode:
        return builtins.open(state["target"], mode, *a, **k)
    return builtins.open(path, mode, *a, **k)


QE.open = safe_open
for t in thrs:
    state["target"] = os.path.join(ROOT, "temp", "gt2", "snap", f"{run}_thr{t}.json")
    QE.cmd_score(SimpleNamespace(run=run, seed=seed, thr=t, full=None, base=None, skip_base_check=True, chord=False, snap=True))
    print("SNAP THR", t, "done", flush=True)

"""delivery threshold of the pair head: the delivered graph with the connectors of p >= thr for several thr (diagnostic on VAL, tuned on VAL: label it so); writes temp/gt2/thr/<run>_thr<t>.json and leaves the run's ends_card.json alone
(the scorer opens ends_card.json for writing itself: the module-level `open` is replaced so that this path goes to the scratch file; the first version only redirected json.dump and truncated the card, 2026-11-12).
usage: thr_sweep.py <run> <seed> <thr> [<thr> ...]"""
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
os.makedirs(os.path.join(ROOT, "temp", "gt2", "thr"), exist_ok=True)
state = {}


def safe_open(path, mode="r", *a, **k):
    if str(path).endswith("ends_card.json") and "w" in mode:
        return builtins.open(state["target"], mode, *a, **k)
    return builtins.open(path, mode, *a, **k)


QE.open = safe_open
for t in thrs:
    state["target"] = os.path.join(ROOT, "temp", "gt2", "thr", f"{run}_thr{t}.json")
    QE.cmd_score(SimpleNamespace(run=run, seed=seed, thr=t, full=None, base=None, skip_base_check=True, chord=False))
    print("THR", t, "done", flush=True)

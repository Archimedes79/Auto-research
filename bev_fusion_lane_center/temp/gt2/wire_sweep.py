"""the wiring radius / angle and the delivery threshold of the SNAPPED read-out (qends_eval.add_connectors(snap=True) through the scorer of record): usage wire_sweep.py <run> <seed> <pos> <ang> <thr> [<thr> ...]; writes temp/gt2/wire/<run>_pos<P>_ang<A>_thr<t>.json, the run's cards are not touched"""
import builtins
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qends_eval as QE

run, seed, pos, ang, thrs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), [float(x) for x in sys.argv[5:]]
QE.WIRE_POS, QE.WIRE_ANG = pos, ang
os.makedirs(os.path.join(ROOT, "temp", "gt2", "wire"), exist_ok=True)
state = {}


def safe_open(path, mode="r", *a, **k):
    if "ends_card" in str(path) and "w" in mode:
        return builtins.open(state["target"], mode, *a, **k)
    return builtins.open(path, mode, *a, **k)


QE.open = safe_open
for t in thrs:
    state["target"] = os.path.join(ROOT, "temp", "gt2", "wire", f"{run}_pos{pos:g}_ang{ang:g}_thr{t}.json")
    QE.cmd_score(SimpleNamespace(run=run, seed=seed, thr=t, full=None, base=None, skip_base_check=True, chord=False, snap=True))
    print("WIRE", pos, ang, t, "done", flush=True)

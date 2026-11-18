"""critic scratch: the parser of the edited module - removed subcommands refused, the remaining ones parse; sweep defaults"""
import importlib.util
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("jtoken_eval", os.path.join(ROOT, "temp", "simplify", "u1_jtoken_eval.py"))
E = importlib.util.module_from_spec(spec)
sys.modules["jtoken_eval"] = E
spec.loader.exec_module(E)
for bad in (["ctl", "--seed", "0", "--ident", "e292_a"], ["fixture", "--seed", "0", "--split", "val"], ["check-parity", "--run", "r", "--seed", "0"], ["sweep", "--run", "r", "--seed", "0", "--split", "train", "--grid", "default"]):
    try:
        E.parse_args(bad)
        print("ACCEPTED", bad)
    except SystemExit:
        print("refused", bad[0], bad[-1])
a = E.parse_args(["sweep", "--run", "r", "--seed", "0", "--split", "train", "--every", "3"])
print("sweep defaults: grid", a.grid, "every", a.every, "max", a.max, "drop", a.drop)
for ok in (["dump", "--run", "r", "--seed", "0"], ["score", "--run", "r", "--seed", "0"], ["ensemble", "--runs", "a,b", "--seed", "0"], ["choose-cell", "--runs", "r"], ["anatomy", "--run", "r", "--seed", "0"],
           ["graphs", "--run", "r", "--seed", "0", "--name", "x", "--out", "y.pkl"], ["live-diff", "--run", "r", "--seed", "0", "--against", "p"], ["check-layer1", "--seed", "0"], ["timing", "--run", "r"]):
    E.parse_args(ok)
print("remaining subcommands parse; PLANB", os.path.relpath(E.PLANB, ROOT), "planb_path", os.path.relpath(E.planb_path(0), ROOT), "exists", os.path.exists(E.planb_path(0)), os.path.exists(E.planb_path(1)))
print("B_DUMP", E.B_DUMP, "exists", os.path.exists(os.path.join(ROOT, E.B_DUMP.format(seed=0))))

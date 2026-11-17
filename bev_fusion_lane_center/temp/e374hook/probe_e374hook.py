"""throw-away: run the scratch test copy against the scratch jtoken copy before they are moved in (deleted afterwards)"""
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", ".."))
for s in ("training/grid", "training", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, s))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


JKs = load("jtoken", os.path.join(HERE, "jtoken_e374hook.py"))
T = load("test_jtoken_scratch", os.path.join(HERE, "test_jtoken_e374hook.py"))
assert T.JK is JKs and T.JKT.JK is JKs
want = sys.argv[1:] or ["test_past", "test_maps", "test_head", "test_arch_options"]
for n in want:
    getattr(T, n)()
print("probe ok")

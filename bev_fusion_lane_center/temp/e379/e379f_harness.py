"""scratch (E379 build): run tests of a scratch test_jtoken against the scratch copies of jtoken / jtoken_train, before they are moved in.

    OMP_NUM_THREADS=2 wsl bash tools/env/_py14.sh temp/e379/e379f_harness.py new [test names]    # e379f_jtoken.py, e379f_jtoken_train.py, e379f_test_jtoken.py
    ... e379f_harness.py base probe                                                                  # the committed copies (e379f_base_*): the champion recipe's 20 smoke steps (the constants)
"""
import json
import os
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
G = os.path.join(ROOT, "training", "grid")
for sub_ in (".", "..", "../data", "../data/prep", "../train", "../../evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(G, sub_)))
pre = "e379f_base_" if sys.argv[1] == "base" else "e379f_"


def load(name, fname, real):
    path = os.path.join(HERE, pre + fname)
    mod = types.ModuleType(name)
    mod.__file__ = real
    sys.modules[name] = mod
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), mod.__dict__)
    mod.__file__ = path                                                       # the textual mutants of the tests read this copy
    return mod


JK = load("jtoken", "jtoken.py", os.path.join(G, "jtoken.py"))
JKT = load("jtoken_train", "jtoken_train.py", os.path.join(ROOT, "training", "train", "jtoken_train.py"))
TJ = load("test_jtoken", "test_jtoken.py", os.path.join(G, "test_jtoken.py"))
TJ.__file__ = os.path.join(G, "test_jtoken.py")
assert JK.__file__.startswith(HERE) and TJ.JK is JK and TJ.JKT is JKT

if sys.argv[2:] == ["probe"]:
    import torch
    losses = []
    real = JKT.make_loss

    def rec(*a, **k):
        f = real(*a, **k)

        def g(m, b):
            r = f(m, b)
            losses.append(float(r[0]))
            return r
        return g
    with tempfile.TemporaryDirectory(dir=TJ.SCRATCH) as runs:
        TJ.patched(JKT, "make_loss", rec, JKT.train, JKT.parse_args(["--name", "c", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0", "--none-weight", "0.25"]), runs, 500, 48)
        row = [json.loads(r) for r in open(os.path.join(runs, "c", "log.jsonl"))][-1]
        sd = torch.load(os.path.join(runs, "c", "model.pt"), map_location="cpu", weights_only=False)["state"]
    print(repr(losses[0]), repr(losses[-1]), repr(row["total"]), repr(sum(float(v.double().abs().sum()) for v in sd.values() if v.dtype.is_floating_point)))
    sys.exit(0)
want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[2:]]
for t_ in TJ.TESTS:
    if not want or t_.__name__ in want:
        t_()
print("harness ok", pre)

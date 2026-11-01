"""throw-away: S8 (compare across sources) and probing on an empty run, against the code given by argv[1] (an autoresearch/ folder)"""
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
from evalsuite import EvalSuite  # noqa: E402
from evalsuite.adapters import toy  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    mk = lambda **kw: EvalSuite(*toy.build(n_scenes=3, n_frames=4, **kw), out=d)  # noqa: E731
    mk().quant.run(name="a", progress=0)
    mk(seed=1).quant.run(name="o", progress=0)
    try:
        r = mk().quant.compare("a", "o", boot=20)
        print("S8: compare ACCEPTED runs of another source (ground truth of seed 1 vs seed 0):", round(r["metrics"]["seg.iou"]["diff"], 4))
    except ValueError as e:
        print("S8: compare refused:", str(e)[:120])
    s = mk()
    s.quant.run(frames=[], name="e", progress=0)
    try:
        print("empty run: situations =", s.probe.situations("e"))
    except Exception as e:  # noqa: BLE001
        print("empty run: situations raised", type(e).__name__, e)

"""smoke test of the statistics part of level_cv with fake per-run numbers (no training): patches train/eval and the loader."""
import sys
sys.path.insert(0, "tests")
import numpy as np
import radar_codecs as rc
rng = np.random.RandomState(0)
rc.rt.load = lambda scenes: [{"x": None}] * 3
rc.eval_codec = lambda c, net, frames: ({"pos (m)": None}, None)
rc.train_codec = lambda *a, **k: object()
base = {"abs": 0.78, "delta": 0.86, "phys": 0.865}
def fake_ok(e, sc):
    return 0.0
n = {"i": 0}
orig = rc._ok_at
cur = {}
def ok_at(e, sc):
    return 0.9 if sc == 1.0 else 0.88
rc._ok_at = ok_at
# make the fake accuracy depend on the codec through eval_codec's return
state = {"name": None}
class C:
    def __init__(self, name): state["name"] = name
rc.Codec = C
def ev(c, net, frames):
    return ({"codec": state["name"], "net": net is not None}, None)
rc.eval_codec = ev
def ok_at2(e, sc):
    if not e["net"]:
        return 0.88 if sc != 1.0 else 0.95
    return base[e["codec"]] + rng.randn() * 0.02 + (0.07 if sc == 1.0 else 0.0)
rc._ok_at = ok_at2
rc.level_cv(1, seeds=(0,), out="temp/radar/cv_smoke.json")

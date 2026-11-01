"""throw-away checks of suspected bugs in evalsuite / trainsuite (review 2026-10-31); run from autoresearch/ with PYTHONPATH=."""
import json
import os
import sys
import tempfile
import time

import numpy as np


def hdr(t):
    print("\n=== " + t, flush=True)


# 1. trainsuite Run.stopped(): operator precedence
hdr("1 trainsuite stopped() after a resume")
from trainsuite.runs import Run
from trainsuite import control
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"steps": 1000}, f)
    control.mark_stopped(d, 100, "kill rule")
    old = time.time() - 3600
    os.utime(os.path.join(d, "STOPPED"), (old, old))            # stopped an hour ago
    with open(os.path.join(d, "log.jsonl"), "w") as f:            # resumed, writing now
        for s in range(100, 160):
            f.write(json.dumps({"step": s, "t": time.time(), "total": 1.0}) + "\n")
    r = Run(d)
    print("STOPPED mtime 1 h old, log written now -> state:", r.state(), "(expected running)")

# 2. trainsuite watch --enforce: stopped_by keyed by rule text across runs
hdr("2 trainsuite watch --enforce with two runs hitting the same rule")
from trainsuite import cli as tcli
with tempfile.TemporaryDirectory() as root:
    for name in ("a", "b"):
        d = os.path.join(root, name)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"steps": 1000}, f)
        with open(os.path.join(d, "log.jsonl"), "w") as f:
            for s in range(0, 600, 10):
                f.write(json.dumps({"step": s, "t": time.time(), "total": 5.0}) + "\n")
    tcli.main(["--root", root, "watch", "a", "b", "--kill", "total@500 > 1", "--enforce", "--once"])
    print("CONTROL in a:", os.path.exists(os.path.join(root, "a", "CONTROL")), " CONTROL in b:", os.path.exists(os.path.join(root, "b", "CONTROL")),
          "(expected True True)")

# 3. kill.txt with an inline comment (as in the README block)
hdr("3 kill.txt with an inline comment")
from trainsuite.kill import load_rules
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "kill.txt"), "w") as f:
        f.write("group.lines@1000 > 1.48              # mean of the last 10 logged values up to step 1000\n")
    try:
        print(load_rules(Run(d)))
    except Exception as e:
        print("load_rules raised:", type(e).__name__, str(e)[:120])

# 3b. crash_reason on a handled traceback followed by normal log lines
hdr("3b crash_reason: traceback printed by a handled exception, training continues")
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "train.log"), "w") as f:
        f.write("it 10 loss 1.0\nTraceback (most recent call last):\n  File x\nOSError: bad file, skipped\nit 11 loss 0.9\nit 12 loss 0.9\n")
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        f.write(json.dumps({"step": 12, "total": 0.9}) + "\n")
    print("state:", Run(d).state(), "(expected running)")

# 4. RunStore: a metric whose frame() returns {} for a whole chunk
hdr("4 RunStore with empty counts of one metric")
from evalsuite.contracts import FrameRef
from evalsuite.store import RunStore
with tempfile.TemporaryDirectory() as d:
    st = RunStore.create(os.path.join(d, "r"), {"name": "r", "state": "reset"})
    for i in range(3):
        st.add(FrameRef("s", i), {"m": {"c": np.ones(2)}, "empty": {}}, {}, {})
    st.close()
    st2 = RunStore.open(os.path.join(d, "r"))
    try:
        print("counts('empty'):", st2.counts("empty"))
    except Exception as e:
        print("counts('empty') raised:", type(e).__name__, str(e)[:100])

# 5. lane_probe cache on a shared LaneGraph object (e.g. a null baseline returning one constant prediction)
hdr("5 lane_probe cache on a reused prediction object")
from evalsuite.contracts import Prediction, Sample
from evalsuite.lanes import Grid, LaneConfig, LaneGraph, LaneTask
cfg = LaneConfig(Grid(u0=-20, v0=-20, res=0.5, nu=80, nv=80))
task = LaneTask(cfg)
EMPTY = LaneGraph()
s1 = Sample(FrameRef("s", 0), None, {"lanes": [np.array([[0.0, -15], [0.0, 15]])]})
s2 = Sample(FrameRef("s", 1), None, {"lanes": [np.array([[-5.0, -15], [-5.0, 15]]), np.array([[5.0, -15], [5.0, 15]])]})
p1 = task.probe(s1, Prediction({"lanes": EMPTY}))
p2 = task.probe(s2, Prediction({"lanes": EMPTY}))
print("frame 0: gt_lanes", p1["tag.gt_lanes"], "gt_len", round(p1["tag.gt_len"], 1), "| frame 1: gt_lanes", p2["tag.gt_lanes"], "gt_len",
      round(p2["tag.gt_len"], 1), "(expected 1 / 30 and 2 / 60)")
causes1 = sum(v for k, v in p1.items() if k.startswith("cause."))
causes2 = sum(v for k, v in p2.items() if k.startswith("cause."))
print("error (sum of causes) frame 0", round(causes1, 1), "frame 1", round(causes2, 1), "(expected about 30 and 60)")

# 6. bootstrap: NaN draws dropped silently
hdr("6 bootstrap_scenes with draws that give NaN")
from evalsuite.stats import bootstrap_scenes


def red(cs, keys=None):
    tp = sum(float(c["tp"]) for c in cs)
    n = sum(float(c["n"]) for c in cs)
    return {"rate": tp / n if n else float("nan")}


a = {f"s{i}": [{"tp": np.array(1.0), "n": np.array(1.0 if i < 2 else 0.0)}] for i in range(10)}     # 8 of 10 scenes have nothing to score
b = {f"s{i}": [{"tp": np.array(0.0), "n": np.array(1.0 if i < 2 else 0.0)}] for i in range(10)}
r = bootstrap_scenes(a, b, red, ["rate"], n=1000, seed=0)
print(r["rate"], "(no count of the NaN draws is reported)")
d = []
rng = np.random.default_rng(0)
for _ in range(1000):
    pick = rng.integers(0, 10, 10)
    d.append(sum(1 for i in pick if i < 2) == 0)
print("share of draws with no scorable scene:", np.mean(d))

# 7. contract checker: reset() check is vacuous without taps
hdr("7 check_adapter on a Subject whose reset() does not forget (no taps)")
from evalsuite.adapters import minimal
from evalsuite import EvalSuite
from evalsuite.check import check_adapter


class Leaky(minimal.MinSubject):
    def __init__(self):
        super().__init__()
        self.mem = 0.0

    def predict(self, sample, taps=()):
        self.mem += 1.0
        return Prediction(outputs=(sample.inputs + self.mem) > self.threshold)

    def reset(self):
        pass                                      # forgets nothing


with tempfile.TemporaryDirectory() as d:
    src, _, tk = minimal.build()
    res = check_adapter(EvalSuite(src, Leaky(), tk, out=d))
    print([(x["name"], x["ok"]) for x in res if "reset" in x["name"]], "(expected ok False)")

# 8. quant trace with the label form the README uses (demo-1/3)
hdr("8 quant trace demo-1/3 (README form) vs demo-1/003")
with tempfile.TemporaryDirectory() as d:
    src, sj, tk = minimal.build()
    suite = EvalSuite(src, sj, tk, out=d)
    suite.quant.run(name="a", progress=0)
    for lab in ("demo-1/3", "demo-1/003"):
        try:
            suite.quant.trace("a", lab, ("overview",))
            print(lab, "ok")
        except Exception as e:
            print(lab, "raised", type(e).__name__, e)

# 9. stats edge cases
hdr("9 robust_z / iqr / loo with one unit")
from evalsuite.stats import loo_influence, robust_z
print("robust_z([1,1,1,1,5]):", robust_z([1, 1, 1, 1, 5]))
print("loo one unit:", loo_influence({"x": [{"tp": np.array(1.0), "n": np.array(1.0)}]}, red))

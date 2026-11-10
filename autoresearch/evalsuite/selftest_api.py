"""Known-answer checks of the class interface (api.py, results.py, apicli.py) on a toy project: 3 scenes of 6 frames, a "network" with a memory (its threshold drifts with the frames it
has seen since the scene started), an IoU metric and a metric that reads a number the network computes itself (a loss, which depends on the ground truth of the moment). No data, no GPU,
seconds:  `python -m evalsuite.selftest_api`  or  `python -m evalsuite Evaluation.selftest`.

What it proves: Frames round-trips and resolves; a ResultSet returns exactly what was stored and belongs to an identity, two writers exclude each other, a dead writer's lock is taken over;
`run` equals the engine's own run (same numbers, carry state) on the kept frames, also for one frame in the middle of a scene (the scene runs whole and warm: D70); a second `run`, and a
subset of an earlier one, take the stored predictions (the network is not called); a new ground truth is scored from them without the network and the self-computed number is flagged; a run
that was killed half-way resumes with the right memory; predict + score = run; `check` finds the problems before anything runs; the reading methods agree with the engine; the command
line prints what the methods return (clean json), maps errors to exit 2, and its words do not collide with the older commands."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import pickle
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from .adapters import minimal
from .api import Evaluation, EvaluationError, Frames
from .contracts import FrameRef, Metric, Prediction, TapSpec
from .results import Locked, ResultSet, result_id, writer_lock
from .suite import EvalSuite

# ------------------------------------------------------------------------------------------------ the toy project (this module is its own adapter)
STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, fail_at=None, memory="ok")
MEMORY_TAP = "memory_conf"


def name(network): return os.path.basename(network)
def network(nm): return nm
def out(net): return os.path.join(STATE.root, name(net))
def sets(): return {"val": 3, "small": 1}
def identity(net): return {"network": net}


class Source(minimal.MinSource):
    def __init__(self, of):
        self.of = of

    def scenes(self):
        return ["demo-0"] if self.of == "small" else super().scenes()

    def load(self, ref):
        STATE.loads += 1
        return super().load(ref)

    def describe(self):
        return {"class": "ToySource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes()), "gt_signature": STATE.gt, "gt_version": None, "evaluator_hash": "x"}


class Subject(minimal.MinSubject):
    """a network with a memory: the threshold falls by 0.04 for every frame seen since reset(); it also computes a 'loss' of its own, which depends on the ground truth of the moment"""
    def __init__(self, base):
        super().__init__(base)
        self.seen = 0

    def reset(self):
        self.seen = 0

    def taps(self):
        return [TapSpec("memory_conf", "map", "memory", "the fill of the memory"), TapSpec("feat", "map", "input", "the input picture")]

    def predict(self, sample, taps=()):
        STATE.calls += 1
        if STATE.fail_at and STATE.calls >= STATE.fail_at:
            raise RuntimeError("killed")
        thr = self.threshold - 0.04 * self.seen
        fill = {"ok": 0.1 * self.seen, "stuck": 0.3, "flat": 0.0}[STATE.memory]      # empty at the start of a scene and growing / full from the start / never filling
        self.seen += 1
        tapped = {"memory_conf": np.full((2, 2), fill), "feat": np.asarray(sample.inputs, np.float32)}
        return Prediction(outputs=sample.inputs > thr, taps={k: v for k, v in tapped.items() if k in taps}, info={"predict_ms": 1.0, "loss": 1.0 if STATE.gt == "g1" else 5.0})

    def describe(self):
        return {"class": "ToySubject", "threshold": self.threshold}


class Loss(Metric):
    name, headline = "loss", ("total",)

    def frame(self, sample, pred):
        return {"v": np.array([float(pred.info["loss"]), 1.0])}

    def reduce(self, counts, keys=None):
        s = sum((c["v"] for c in counts), np.zeros(2))
        return {"total": float(s[0] / max(s[1], 1.0))}

    def direction(self, key):
        return "low"


class Task(minimal.MinTask):
    def metrics(self):
        return [minimal.IoU(), Loss()]


def source(net, of, workers=0, control=""): return Source(of)
def subject(net, src, device): return Subject(float(net.split("thr")[-1]))


def task(net, metrics):
    if set(metrics) - {"maps", "boxes"}:
        raise ValueError(f"unknown metrics {sorted(set(metrics) - {'maps', 'boxes'})}")
    return Task()


def reference(src, ref):
    truth = np.asarray(minimal.MinSource.load(src, ref).reference)
    return np.roll(truth, STATE.shift, axis=1) if STATE.shift else truth          # a "new ground truth": the disc moved


def card(info, store, quant, rs):
    return {"name": info["id"], "metrics": quant.table(store)}


ME = sys.modules[__name__]


# ------------------------------------------------------------------------------------------------ the checks
def _raises(exc, text, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"no {exc.__name__} with {text!r}")


def _engine(net, frames_spec):
    """the numbers of the engine's own run: the scenes whole, state carry, scored on the kept frames only"""
    src, subj = minimal.MinSource(), Subject(float(net.split("thr")[-1]))
    suite = EvalSuite(src, subj, Task(), out=tempfile.mkdtemp(), state="carry")
    keep = frames_spec.resolve(src)[1]
    store = suite.quant.run(scenes=sorted({r.scene for r in keep}), name="eng", progress=0)
    return suite.quant.table(store, [r.label for r in keep])


def _same(got, want, what):
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-12 for k in want), (what, got, want)


def check_frames():
    src = Source("val")
    for text in ("val", "small.val", "spread:5:val", "random:2:7:small.val", "scenes:demo-0,demo-1:val", "frames:demo-0/1,demo-2/5:val"):
        assert str(Frames.parse(text)) == text, text
    assert Frames.parse("random:3:small.val") == Frames.random(3, 0, "small.val") and Frames.parse("random:3") == Frames.random(3, 0) and Frames.parse("random:3:7") == Frames.random(3, 7)
    assert Frames.parse("random:3:7:small.val").args == (3, 7) and str(Frames.parse("random:3")) == "random:3:0:val", "the seed is optional and shown when it is not"
    assert Frames.parse("spread:9:small.val").of == "small.val" and Frames.parse("scenes:a,b").args == (("a", "b"),)
    assert Frames.parse("spread:5") == Frames.spread(5) and str(Frames.parse("spread:5")) == "spread:5:val" and str(Frames.parse(Frames.all("x"))) == "x"
    scenes, keep = Frames.all().resolve(src)
    assert scenes == ["demo-0", "demo-1", "demo-2"] and len(keep) == 18
    s5, k5 = Frames.spread(5).resolve(src)
    assert len(k5) == 5 and k5 == sorted(k5) and s5 == sorted({r.scene for r in k5})
    a, b = Frames.random(2, 1).resolve(src)[0], Frames.random(2, 1).resolve(src)[0]
    assert a == b and len(a) == 2 and any(Frames.random(2, s).resolve(src)[0] != a for s in range(2, 8)), "random: reproducible with its seed, another seed another scenes"
    assert Frames.random(2, 1).resolve(src)[1] == [r for s in a for r in src.frames(s)], "random scenes keep all their frames"
    assert [r.label for r in Frames.pick(["demo-2/5", "demo-0/1"]).resolve(src)[1]] == ["demo-0/001", "demo-2/005"]
    for bad in ("spread:x", "random", "random:x", "scenes", "frames:a:b:c:d", "spread:1:2:3", "random:3:7:small.val:x"):
        _raises(EvaluationError, "", Frames.parse, bad)
    _raises(EvaluationError, "not in val", Frames.scenes(["demo-9"]).resolve, src)
    _raises(EvaluationError, "not a frame", Frames.pick(["demo-0/99"]).resolve, src)
    _raises(EvaluationError, "not a frame", Frames.pick(["junk"]).resolve, src)
    _raises(EvaluationError, "takes", Frames, "spread", ())


def check_results():
    tmp = tempfile.mkdtemp()
    refs = minimal.MinSource().frames("demo-1")[:3]
    ident = {"network": {"name": "a"}, "inputs": {"x": 1}}
    rid = result_id(ident)
    assert rid == result_id({"inputs": {"x": 1}, "network": {"name": "a"}}) and rid != result_id({**ident, "inputs": {"x": 2}}), "the id is the identity, whatever the key order"
    _raises(ValueError, "named by its id", ResultSet.create, os.path.join(tmp, "wrong"), ident)
    rs = ResultSet.create(os.path.join(tmp, rid), ident)
    assert rs.refs() == [] and not rs.has(refs[0])
    pred = Prediction(outputs={"dt": {0.0: {"prob": {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, "dets": [{"x": 1.5, "c": "car"}], "none": None}}},
                      taps={"t": np.ones(2)}, info={"predict_ms": 2.5})
    rs.add(refs[0], pred)
    back = rs.load(refs[0])
    assert np.array_equal(back.outputs["dt"][0.0]["prob"]["a"], pred.outputs["dt"][0.0]["prob"]["a"]) and back.outputs["dt"][0.0]["dets"] == [{"x": 1.5, "c": "car"}]
    assert back.outputs["dt"][0.0]["none"] is None and set(back.outputs["dt"]) == {0.0} and back.info == {"predict_ms": 2.5} and back.taps == {} and np.array_equal(rs.load(refs[0], ("t",)).taps["t"], np.ones(2)), "keys and types survive; taps come on request"
    assert rs.has(refs[0]) and not rs.has(refs[1])
    _raises(KeyError, "not in result set", rs.load, refs[1])
    for r in refs[1:]:
        rs.add(r, pred)
    rs.close(wall_s=1.0, cost={"median_ms": 2.5})
    again = ResultSet.open(os.path.join(tmp, rid))
    assert [r.label for r in again.refs()] == [r.label for r in refs] and again.manifest["format"] == 1 and again.n_frames() == 3
    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"
    # taps are stored next to the frame; a frame without a tap does not have it, adding a tap keeps the outputs
    assert not again.has(refs[1], ("feat",)) and again.tap_names(refs[1]) == ["t"]
    rw = ResultSet.open(os.path.join(tmp, rid), "a")
    rw.add(refs[1], Prediction(outputs="other", taps={"feat": np.arange(4.0).reshape(2, 2), "a/b": np.ones(1)}, info={}))
    got = ResultSet.open(os.path.join(tmp, rid)).load(refs[1], ("feat", "a/b"))
    assert rw.has(refs[1], ("feat",)) and not rw.has(refs[1], ("nope",)) and np.array_equal(got.taps["feat"], np.arange(4.0).reshape(2, 2)) and got.outputs["dt"][0.0]["dets"] == [{"x": 1.5, "c": "car"}]
    assert rw.tap_names(refs[1]) == ["a_b", "feat", "t"], "outputs of a stored frame are kept, a second add only adds taps"
    # the dataset key is stored with the frame: another key at the same index (a rebuilt data set) is an error, not a wrong score
    keyed = [FrameRef("demo-9", i, f"tok{i}") for i in range(2)]
    rk = ResultSet.create(os.path.join(tmp, rid), ident)
    rk.add(keyed[0], pred)
    assert rk.load(FrameRef("demo-9", 0, "tok0")).info == {"predict_ms": 2.5} and rk.load(FrameRef("demo-9", 0)).info, "the same key, or none, reads"
    _raises(IOError, "the dataset changed", rk.load, FrameRef("demo-9", 0, "other"))
    old = json.load(open(os.path.join(tmp, rid, "manifest.json")))
    old["format"] = 0
    json.dump(old, open(os.path.join(tmp, rid + "_old.json"), "w"))
    os.makedirs(os.path.join(tmp, "oldset"))
    json.dump(old, open(os.path.join(tmp, "oldset", "manifest.json"), "w"))
    _raises(ValueError, "result set format 0", ResultSet.open, os.path.join(tmp, "oldset"))
    _raises(KeyError, "with the taps", rw.load, refs[2], ("feat",))
    _raises(IOError, "read-only", again.add, refs[0], pred)
    assert not [f for f in os.listdir(os.path.join(tmp, rid, "frames")) if ".tmp" in f], "no half-written file stays"
    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.pkl"), "wb") as f:
        f.write(b"not a frame")
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[0])
    # two writers: the second is refused while the first holds the lock; a dead writer's lock is taken over; the lock goes away with its holder
    target = os.path.join(tmp, "folder")
    with writer_lock(target):
        _raises(Locked, "being written by pid", writer_lock(target).__enter__)
    assert not os.path.exists(target + ".lock")
    with open(target + ".lock", "w") as f:
        f.write(f"4194301 {socket.gethostname()} deadbeef")
    with writer_lock(target):
        pass
    assert not os.path.exists(target + ".lock")
    with open(target + ".lock", "w") as f:
        f.write("4194301 another-host deadbeef")
    _raises(Locked, "another-host", writer_lock(target).__enter__)
    os.remove(target + ".lock")
    with open(target + ".lock", "w") as f:                                      # an empty lock: a writer that is just creating it is waited for, a damaged one (old) is taken over
        pass
    os.utime(target + ".lock", (1, 1))
    with writer_lock(target):
        pass
    with writer_lock(target):                                                   # a holder removes only a lock that carries its own token
        with open(target + ".lock", "w") as f:
            f.write(f"{os.getpid()} {socket.gethostname()} someone-else")
    assert os.path.exists(target + ".lock") and "someone-else" in open(target + ".lock").read()
    os.remove(target + ".lock")


def check_run():
    STATE.root, STATE.gt, STATE.shift, STATE.calls, STATE.loads, STATE.fail_at = tempfile.mkdtemp(), "g1", 0, 0, 0, None
    ev = Evaluation(adapter=ME)
    net = "runs/thr0.5"
    # the engine's own run is the reference: same numbers on the kept frames (the kept frame of one scene in the middle runs with the scene's history)
    for spec in ("val", "frames:demo-1/4", "scenes:demo-2:val", "spread:7"):
        fr = Frames.parse(spec)
        r = ev.run(net, spec, gt_key="g1", progress=0)
        _same(r.quant()["numbers"], _engine(net, fr), spec)
        assert r["n_frames"] == len(fr.resolve(Source("val"))[1]) and r["warnings"] == []
    one, allr = ev.run(net, "frames:demo-1/4", progress=0), ev.run(net, "val", progress=0)
    assert one.by("frame")["demo-1/004"]["iou.iou"] == allr.by("frame")["demo-1/004"]["iou.iou"], "a kept frame in the middle of a scene: the scene ran whole and warm"
    # stored predictions are reused: the same question, and a SUBSET of an earlier one (spread:7 and frames:demo-1/4 are inside val), call the network not at all
    STATE.calls = 0
    r2, sub = ev.run(net, "val", progress=0), ev.run(net, "scenes:demo-0:val", progress=0)
    assert STATE.calls == 0 and r2["predictions_reused"] and r2["id"] == allr["id"] and r2.quant() == allr.quant(), "the same question"
    assert sub["predictions_reused"] and sub["results"] == allr["results"] and sub["id"] != allr["id"] and sub["n_frames"] == 6, "a subset costs no network"
    # predict + score = run
    pr = ev.predict(net, "val")
    assert pr["results"] == allr["results"] and pr["frames_predicted"] == 0 and ev.score(pr["results"], "val")["id"] == allr["id"]
    _raises(EvaluationError, "not in", ev.score, pr["results"], "scenes:demo-9")
    # a new ground truth: another evaluation id, scored from the stored predictions, no network and no input loaded; the loss the network computed itself is flagged
    STATE.gt, STATE.shift, STATE.calls, STATE.loads = "g2", 3, 0, 0
    r3 = ev.run(net, "val", gt_key="g2", progress=0)
    assert STATE.calls == 0 and STATE.loads == 0 and r3["predictions_reused"] and r3["results"] == allr["results"] and r3["id"] != allr["id"], "rescored without the network"
    assert r3.quant()["numbers"]["iou.iou"] != allr.quant()["numbers"]["iou.iou"], "the other ground truth gives other numbers"
    assert len(r3["warnings"]) == 1 and "ground truth" in r3["warnings"][0] and r3.quant()["numbers"]["loss.total"] == allr.quant()["numbers"]["loss.total"] == 1.0, "the stale loss is said"
    STATE.gt, STATE.shift = "g1", 0
    _raises(EvaluationError, "has ground truth g1", ev.run, net, "val", gt_key="g9")
    STATE.gt = None
    _raises(EvaluationError, "no ground-truth signature", ev.run, net, "val")
    STATE.gt = "g1"
    _raises(EvaluationError, "not in val", ev.run, net, "scenes:demo-7")
    _raises(EvaluationError, "unknown metrics", ev.run, net, "val", metrics=("mapz",))
    # another network = other predictions and another id; the sign of compare is variant - control
    low = ev.run("runs/thr0.1", "val", progress=0)
    assert low["results"] != allr["results"] and low["id"] != allr["id"]
    c = low.compare(allr)
    assert c["metrics"]["iou.iou"]["diff"] == low.quant()["numbers"]["iou.iou"] - allr.quant()["numbers"]["iou.iou"], "compare is variant - control"
    assert ev.compare(allr["id"], low["id"])["metrics"]["iou.iou"]["diff"] == c["metrics"]["iou.iou"]["diff"]
    # a run killed half-way resumes from the start of its scene, so the memory is right: the numbers are the engine's
    net3, STATE.calls, STATE.fail_at = "runs/thr0.3", 0, 9
    _raises(RuntimeError, "killed", ev.run, net3, "val", progress=0)
    stored = [x for x in ev.list(net3) if x["kind"] == "predictions"]
    assert len(stored) == 1 and stored[0]["frames"] == 8, stored
    STATE.fail_at = None
    done = ev.run(net3, "val", progress=0)
    _same(done.quant()["numbers"], _engine(net3, Frames.all()), "resumed run")
    # a lock held by a live writer refuses the second one with a message
    net4 = "runs/thr0.7"
    rid = ev.check(net4, "val")["results"].partition("@")[2]
    os.makedirs(os.path.join(out(net4), "results"), exist_ok=True)
    with open(os.path.join(out(net4), "results", rid + ".lock"), "w") as f:
        f.write(f"{os.getpid()} {socket.gethostname()} tok")
    _raises(EvaluationError, "being written by pid", ev.run, net4, "val", progress=0)
    os.remove(os.path.join(out(net4), "results", rid + ".lock"))
    # intermediate results: asked for with the predictions; a frame stored without them is predicted again, one with them is reused; unknown taps are refused before the network runs
    net5, STATE.calls = "runs/thr0.6", 0
    plain = ev.predict(net5, "scenes:demo-0:val")
    assert plain["frames_predicted"] == 6
    with_tap = ev.predict(net5, "scenes:demo-0:val", taps=("feat",))
    assert with_tap["frames_predicted"] == 6 and with_tap["results"] == plain["results"], "stored without the tap: predicted again to get it"
    STATE.calls = 0
    assert ev.predict(net5, "scenes:demo-0:val", taps=("feat",))["frames_predicted"] == 0 and STATE.calls == 0 and ev.predict(net5, "scenes:demo-0:val")["frames_predicted"] == 0, "and reused"
    assert ev.tap(plain["results"], "demo-0/2")["taps"] == ["feat"]
    t = ev.tap(plain["results"], "demo-0/2", "feat")
    assert t["shape"] == [32, 32] and np.load(t["path"]).shape == (32, 32)
    _raises(EvaluationError, "has no tap", ev.tap, plain["results"], "demo-0/2", "memory_conf")
    _raises(EvaluationError, "unknown taps", ev.predict, net5, "val", taps=("nothing",))
    # the health of the memory (D70): healthy = no warning; full from the start, or never filling = a warning in the predictions' result
    assert plain["warnings"] == [] and ev.run(net5, "scenes:demo-0:val", progress=0)["warnings"] == []
    for mode, text in (("stuck", "not empty at the first frame"), ("flat", "does not fill")):
        STATE.memory = mode
        r = ev.run(f"runs/thr0.{4 if mode == 'stuck' else 2}", "scenes:demo-1:val", progress=0)
        assert len(r["warnings"]) == 1 and text in r["warnings"][0] and "demo-1" in r["warnings"][0], (mode, r["warnings"])
    STATE.memory = "ok"
    # the dry run: what would happen, and the problems before anything is run
    chk = ev.check(net, "frames:demo-1/4")
    assert chk["ok"] and chk["scenes_run"] == 1 and chk["frames_kept"] == 1 and chk["frames_stored"] == 1 and chk["network_runs"] is False and chk["id"].startswith("thr0.5@e")
    assert chk["frames_to_run"] == 0 and ev.check("runs/thr0.9", "frames:demo-1/4")["frames_to_run"] == 5, "frames 0..4 of the scene run (the memory), nothing after the kept one"
    assert ev.check("runs/thr0.9", "val")["frames_to_run"] == 18
    nosuch = ev.check(net, "nosuch")
    assert not nosuch["ok"] and "not one of" in nosuch["problems"][0] and "TEST" in ev.check(net, "test")["problems"][0]
    _raises(EvaluationError, "not one of", ev.run, net, "nosuch")
    assert ev.check("runs/thr0.9", "spread:3")["network_runs"] is True
    bad = ev.check(net, "scenes:demo-9", metrics=("mapz",))
    assert not bad["ok"] and bad["problems"], bad
    assert not ev.check(net, "frames:demo-1/4", metrics=("mapz",))["ok"]
    STATE.gt = None
    assert not ev.check(net, "val")["ok"]
    STATE.gt = "g1"
    # the result is a plain dict when copied or pickled
    assert type(copy.deepcopy(allr)) is dict and pickle.loads(pickle.dumps(allr)) == dict(allr)
    return ev, net, allr, low


def check_reading(ev, net, allr, low):
    q = allr.quant("iou")
    assert q["n_frames"] == 18 and set(q["numbers"]) >= {"iou.iou"} and q["better"]["iou.iou"] == "high" and all("iou" in k for k in q["numbers"])
    assert set(allr.by("scene")) == {"demo-0", "demo-1", "demo-2"} and len(allr.by("frame")) == 18
    _raises(EvaluationError, "axis", allr.by, "range")
    assert set(allr.worst(n=2)) == {"scenes", "frames", "flags"}
    assert allr.cost()["predict"]["n"] == 18 and allr.cost()["devices"] == ["cuda"]
    assert os.path.isfile(allr.report()) and allr.card()["name"] == allr["id"]
    rows = ev.list(net)
    assert {r["kind"] for r in rows} == {"predictions", "evaluation"} and allr["id"] in {r["id"] for r in rows} and allr["results"] in {r["id"] for r in rows}
    assert all(r["megabytes"] >= 0 for r in rows if r["kind"] == "predictions")
    assert ev.sets() == {"val": 3, "small": 1}
    _raises(EvaluationError, "not an evaluation id", ev.quant, allr["results"])
    _raises(EvaluationError, "no evaluation", ev.quant, "thr0.5@edeadbeef0000")
    assert Evaluation(adapter=ME).quant(allr["id"])["numbers"] == allr.quant()["numbers"], "a new object finds the evaluation by its id"
    pic = ev.qual_frame(allr["id"], "demo-1/4")
    assert os.path.isfile(pic["images"]["overview"]) and "iou.iou" in pic["scores"]


def check_cli():
    from . import cli
    from .apicli import LEGACY, _clean, dispatch
    base = ["--adapter", __name__, "--project", STATE.root]

    def call(*argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = dispatch(list(argv))
        return code, buf.getvalue(), err.getvalue()
    code, text, _ = call("Evaluation.run", "runs/thr0.5", "val", "--json", "--workers", "0", "--progress", "0", *base)
    info = json.loads(text)                                                                   # stdout is the json and nothing else
    assert code == 0 and info["frames"] == "val" and info["predictions_reused"] is True and info["id"].startswith("thr0.5@e"), (code, text[:200])
    code, text, _ = call("Evaluation.quant", info["id"], "--match", "iou", "--json", *base)
    assert code == 0 and json.loads(text)["numbers"]["iou.iou"] == Evaluation(adapter=ME).quant(info["id"])["numbers"]["iou.iou"]
    assert call("quant", info["id"], *base)[0] is None and call("probe", "x", *base)[0] is None and call("list", "x", *base)[0] is None, "older command words are not taken over"
    code, text, _ = call("check", "runs/thr0.5", "val", *base)
    assert code == 0 and "ok: True" in text and "frames_stored: 18" in text, text
    code, text, _ = call("qual-frame", info["id"], "demo-0/2", *base)                         # the hyphen form of qual_frame
    assert code == 0 and ".png" in text, text
    code, text, _ = call("methods")
    assert code == 0 and "Evaluation.run(" in text and "Evaluation.compare(" in text and "gt_key=None" in text
    code, _, err = call("Evaluation.quant", "thr0.5@e0000", *base)
    assert code == 2 and "no evaluation" in err
    code, _, err = call("run", "runs/thr0.5", "scenes:demo-9", *base)                         # a data error is a message and exit 2, not a traceback
    assert code == 2 and "not in val" in err and "Traceback" not in err, err
    code, _, err = call("run", "runs/thr0.5", "val", "--metrics", "mapz", *base)
    assert code == 2 and "unknown metrics" in err, err
    code, text, _ = call("check", "runs/thr0.5", "scenes:demo-9", *base)
    assert code == 1 and "ok: False" in text, text
    assert call("nothing")[0] is None
    assert _clean({"a": float("nan"), "b": [np.float32(1.5), float("inf")], "c": (1, 2)}) == {"a": None, "b": [1.5, None], "c": [1, 2]}
    # the words of the older command line stay theirs: a method with the name of an old command must be listed in LEGACY
    old = {c for a in cli.build_parser()._actions if isinstance(a, __import__("argparse")._SubParsersAction) for c in a.choices}
    assert {m for m in Evaluation.METHODS if m in old} <= set(LEGACY), (old, Evaluation.METHODS)


def run_all():
    check_frames()
    check_results()
    ev, net, allr, low = check_run()
    check_reading(ev, net, allr, low)
    check_cli()
    shutil.rmtree(STATE.root, ignore_errors=True)
    return {"ok": True, "checks": 5, "what": "frames, result set and locks, run = engine on the kept frames (warm), reuse / subset / rescoring / resume / predict+score / check, reading methods, command line"}


if __name__ == "__main__":
    print(json.dumps(run_all()))

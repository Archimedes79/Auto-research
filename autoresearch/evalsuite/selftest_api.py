"""Known-answer checks of the class interface (api.py, results.py, apicli.py) on a toy project: 3 scenes of 6 frames, a "network" with a memory (its threshold drifts with the frames it
has seen since the scene started) and an IoU metric; its Runner side (free_source / free_subject) reads no ground truth. No data, no GPU,
seconds:  `python -m evalsuite.selftest_api`  or  `python -m evalsuite Evaluation.selftest`.

What it proves: Frames round-trips and resolves; a ResultSet returns exactly what was stored and belongs to an identity, two writers exclude each other, a dead writer's lock is taken over;
`run` equals the engine's own run (same numbers, carry state) on the kept frames, also for one frame in the middle of a scene (the scene runs whole and warm: D70); a second `run`, and a
subset of an earlier one, take the stored predictions (the network is not called); a new ground truth is scored from them without the network and the self-computed number is flagged; a run
that was killed half-way resumes with the right memory; Runner.predict + score = Evaluation.run; `check` finds the problems before anything runs; the reading methods agree with the engine; the command
line prints what the methods return (clean json), maps errors to exit 2, and its words do not collide with the older commands."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np

from .adapters import minimal
from .api import ADAPTER_FUNCTIONS, Evaluation, EvaluationError, Frames, load_adapter
from .runner_api import Runner
from .contracts import FrameRef, Prediction, Sample, TapSpec
from . import results as results_mod
from .results import Locked, ResultSet, decode, encode, result_id, writer_lock
from .suite import EvalSuite

# ------------------------------------------------------------------------------------------------ the toy project (this module is its own adapter)
STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False, ev="x")
MEMORY_TAP = "memory_conf"
COMPARE_IGNORE = {"source": ("evaluator_hash",)}                                   # the arms may differ in the evaluator code: compare says so in its warnings


def name(network): return os.path.basename(network)
def network(nm): return nm
def out(net): return os.path.join(STATE.root, name(net))
def sets(): return {"val": 3, "small": 1}
def identity(net): return {"network": net, "checkpoint": STATE.bump}


class Source(minimal.MinSource):
    def __init__(self, of):
        self.of = of

    def scenes(self):
        return ["demo-0"] if self.of == "small" else super().scenes()

    def load(self, ref):
        STATE.loads += 1
        return super().load(ref)

    def describe(self):
        return {"class": "ToySource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes()), "gt_signature": STATE.gt, "gt_version": None, "evaluator_hash": STATE.ev}


class Subject(minimal.MinSubject):
    """a network with a memory: the threshold falls by 0.04 for every frame seen since reset()"""
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
        return Prediction(outputs=sample.inputs > thr, taps={k: v for k, v in tapped.items() if k in taps}, info={"predict_ms": 1.0})

    def describe(self):
        return {"class": "ToySubject", "threshold": self.threshold}


class FreeSource(minimal.MinSource):
    """the data the Runner reads: no ground truth (reference None), its own load counter; it describes what the network is fed like the labelled source does"""
    def __init__(self, of, control=""):
        self.of, self.control = of, control

    def scenes(self):
        return ["demo-0"] if self.of == "small" else super().scenes()

    def load(self, ref):
        STATE.free_loads += 1
        s = super().load(ref)
        return Sample(s.ref, inputs=s.inputs, reference=None, meta={"token": ref.key})

    def describe(self):
        return {"class": "FreeSource", "dataset": "toy", "split": self.of, "scenes": len(self.scenes()), "control": self.control or None}


def source(net, of, workers=0, control=""): return Source(of)                       # with ground truth: what the Evaluation scores with
def free_source(net, of, workers=0, control=""): return FreeSource(of, control)       # without: what the Runner runs on
def subject(net, src, device): return Subject(float(net.split("thr")[-1]))
def free_subject(net, src, device):                                                  # (the toy network never reads a ground truth)
    STATE.bump += 1 if STATE.change else 0                                           # "a training wrote a new checkpoint while the network was loaded"
    return subject(net, src, device)


def task(net, metrics):
    if set(metrics) - {"maps", "boxes"}:
        raise ValueError(f"unknown metrics {sorted(set(metrics) - {'maps', 'boxes'})}")
    return minimal.MinTask()


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
    suite = EvalSuite(src, subj, minimal.MinTask(), out=tempfile.mkdtemp(), state="carry")
    keep = frames_spec.resolve(src)[1]
    store = suite.quant.run(scenes=sorted({r.scene for r in keep}), name="eng", progress=0)
    return suite.quant.table(store, [r.label for r in keep])


def _same(got, want, what):
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-12 for k in want), (what, got, want)


def check_frames():
    src = Source("val")
    for text in ("val", "small.val", "spread:5:val", "random:2:7:small.val", "scenes:demo-0,demo-1:val", "frames:demo-0/1,demo-2/5:val"):
        assert str(Frames.parse(text)) == text, text
    assert Frames.parse("random:3:small.val").args == (3, 0) and Frames.parse("random:3").args == (3, 0) and Frames.parse("random:3:7").args == (3, 7)
    assert Frames.parse("random:3:7:small.val").args == (3, 7) and str(Frames.parse("random:3")) == "random:3:0:val", "the seed is optional and shown when it is not"
    assert Frames.parse("spread:9:small.val").of == "small.val" and Frames.parse("scenes:a,b").args == (("a", "b"),)
    assert str(Frames.parse("spread:5")) == "spread:5:val" and str(Frames.parse(Frames.parse("x"))) == "x"
    P = Frames.parse
    scenes, keep = P("val").resolve(src)
    assert scenes == ["demo-0", "demo-1", "demo-2"] and len(keep) == 18
    s5, k5 = P("spread:5").resolve(src)
    assert len(k5) == 5 and k5 == sorted(k5) and s5 == sorted({r.scene for r in k5})
    a, b = P("random:2:1").resolve(src)[0], P("random:2:1").resolve(src)[0]
    assert a == b and len(a) == 2 and any(P(f"random:2:{s}").resolve(src)[0] != a for s in range(2, 8)), "random: reproducible with its seed, another seed another scenes"
    assert P("random:2:1").resolve(src)[1] == [r for s in a for r in src.frames(s)], "random scenes keep all their frames"
    assert [r.label for r in P("frames:demo-2/5,demo-0/1").resolve(src)[1]] == ["demo-0/001", "demo-2/005"]
    for bad in ("spread:x", "random", "random:x", "scenes", "frames:a:b:c:d", "spread:1:2:3", "random:3:7:small.val:x"):
        _raises(EvaluationError, "", Frames.parse, bad)
    _raises(EvaluationError, "not in val", P("scenes:demo-9").resolve, src)
    _raises(EvaluationError, "not a frame", P("frames:demo-0/99").resolve, src)
    _raises(EvaluationError, "not a frame", P("frames:junk").resolve, src)
    _raises(EvaluationError, "takes", Frames, "spread", ())


def check_codec():
    """encode / decode: everything a prediction holds comes back with its types; nothing that needs pickle is stored"""
    obj = {"dt": {-0.5: {"prob": {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, "dets": [{"x": np.float32(1.5), "n": 3, "c": "car", "ok": True, "box": (1.0, 2.0)}], "none": None},
                  0.0: {"vhat": np.zeros((2, 2), np.float16)}, 1: {"i": np.int64(7), "m": np.array([True, False])}},
           "nan": float("nan"), "inf": float("-inf"), "scalar": np.array(2.5), "empty": np.zeros((0, 3), np.float32), "ints": [1, 2, [3]]}
    tree, arrays = encode(obj)
    json.dumps(tree, allow_nan=False)                                          # strict json: no bare NaN
    back = decode(json.loads(json.dumps(tree)), arrays)
    assert list(back["dt"]) == [-0.5, 0.0, 1] and [type(k) for k in back["dt"]] == [float, float, int], "keys keep their type and order"
    p = back["dt"][-0.5]
    assert np.array_equal(p["prob"]["a"], obj["dt"][-0.5]["prob"]["a"]) and p["prob"]["a"].dtype == np.float32 and p["none"] is None
    d = p["dets"][0]
    assert d == {"x": np.float32(1.5), "n": 3, "c": "car", "ok": True, "box": (1.0, 2.0)} and type(d["x"]) is np.float32 and type(d["box"]) is tuple and d["ok"] is True
    assert back["dt"][0.0]["vhat"].dtype == np.float16 and back["dt"][1]["m"].dtype == bool and type(back["dt"][1]["i"]) is np.int64
    assert back["nan"] != back["nan"] and back["inf"] == float("-inf") and back["scalar"].shape == () and back["empty"].shape == (0, 3) and back["ints"] == [1, 2, [3]]
    _raises(TypeError, "object array", encode, {"a": np.array([None, 1], dtype=object)})
    _raises(TypeError, "cannot store", encode, {"a": {1, 2}})
    _raises(TypeError, "cannot store", encode, {"a": object()})


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
    assert [r.label for r in again.refs()] == [r.label for r in refs] and again.manifest["format"] == 2 and again.n_frames() == 3
    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"
    # the frame file is a plain npz that any numpy reads (no pickle): the arrays and tree.json
    with np.load(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), allow_pickle=False) as z:
        assert "tree.json" in z.files and "a0" in z.files and np.array_equal(z["a0"], np.arange(6, dtype=np.float32).reshape(2, 3))
        assert "$d" in json.loads(bytes(z["tree.json"]).decode("utf-8"))
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
    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.npz"), "wb") as f:
        f.write(b"not a frame")
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[0])
    good = open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "rb").read()
    mid = bytearray(good)
    with zipfile.ZipFile(io.BytesIO(good)) as z:
        off = z.getinfo("tree.json").header_offset
    nlen, elen = struct.unpack("<HH", good[off + 26: off + 30])
    mid[off + 30 + nlen + elen + 1] ^= 0xFF                                             # a byte inside the compressed json: a zlib / CRC error, not a traceback
    with open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "wb") as f:
        f.write(bytes(mid))
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[1])
    with open(os.path.join(tmp, rid, "frames", "notes.npz"), "wb") as f:                # a file that is not named <scene>_<index>.npz is not a frame
        f.write(b"x")
    listed = ResultSet.open(os.path.join(tmp, rid))
    assert {r.label for r in listed.refs()} == {r.label for r in refs} | {"demo-9/000"} and listed.n_frames() == 4, "notes.npz is not a frame"
    # package.py clean removes the emptied frames / taps folders and keeps the manifest: the set takes frames again
    shutil.rmtree(os.path.join(tmp, rid, "frames"))
    shutil.rmtree(os.path.join(tmp, rid, "taps"))
    empty = ResultSet.open(os.path.join(tmp, rid), "a")
    assert empty.n_frames() == 0 and not empty.has(refs[0])
    empty.add(refs[0], Prediction(outputs="o", taps={"t": np.ones(1)}, info={}))
    assert empty.has(refs[0], ("t",)) and empty.load(refs[0], ("t",)).outputs == "o"
    # two writers: the second is refused while the first holds the (flock) lock, and says who holds it; the kernel drops the lock with its holder, so a dead writer's file does not block
    target = os.path.join(tmp, "folder")
    if results_mod.fcntl is None:
        print("  (no fcntl here: the writer lock is not tested)")
        return
    with writer_lock(target):
        _raises(Locked, "being written by pid", writer_lock(target).__enter__)
    with writer_lock(target):                                                   # released: the next writer gets it; the lock file stays, its content is only the message
        pass
    with open(target + ".lock", "w") as f:                                      # a lock file left by a dead writer does not block
        f.write("pid 4194301 on another-host")
    with writer_lock(target):
        pass
    code = ("import sys; sys.path.insert(0, %r); from evalsuite.results import writer_lock; import time\n"
            "with writer_lock(%r):\n    print('held', flush=True); time.sleep(30)\n") % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), target)
    other = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)   # a second PROCESS holds it: this one is refused with the holder's pid
    try:
        assert other.stdout.readline().strip() == "held"
        _raises(Locked, f"pid {other.pid}", writer_lock(target).__enter__)
    finally:
        other.kill()
        other.wait()
    with writer_lock(target):                                                   # killed holder: free at once
        pass


def check_run():
    STATE.root, STATE.gt, STATE.shift, STATE.calls, STATE.loads, STATE.free_loads, STATE.fail_at = tempfile.mkdtemp(), "g1", 0, 0, 0, 0, None
    ev = Evaluation(adapter=ME)
    net = "runs/thr0.5"
    # the engine's own run is the reference: same numbers on the kept frames (the kept frame of one scene in the middle runs with the scene's history)
    for spec in ("val", "frames:demo-1/4", "scenes:demo-2:val", "spread:7"):
        fr = Frames.parse(spec)
        r = ev.run(net, spec, gt_key="g1", progress=0)
        _same(ev.show_numbers(r["id"])["numbers"], _engine(net, fr), spec)
        assert r["n_frames"] == len(fr.resolve(Source("val"))[1]) and r["warnings"] == []
    assert STATE.free_loads > 0 and STATE.loads == 0, "the Runner reads the frames through free_source; the labelled Source.load is never called (no ground truth, no inputs of the labelled data)"
    one, allr = ev.run(net, "frames:demo-1/4", progress=0), ev.run(net, "val", progress=0)
    assert ev.split_numbers(one["id"], "frame")["demo-1/004"]["iou.iou"] == ev.split_numbers(allr["id"], "frame")["demo-1/004"]["iou.iou"], "a kept frame in the middle of a scene: the scene ran whole and warm"
    # stored predictions are reused: the same question, and a SUBSET of an earlier one (spread:7 and frames:demo-1/4 are inside val), call the network not at all
    STATE.calls = 0
    r2, sub = ev.run(net, "val", progress=0), ev.run(net, "scenes:demo-0:val", progress=0)
    assert STATE.calls == 0 and r2["predictions_reused"] and r2["id"] == allr["id"] and ev.show_numbers(r2["id"]) == ev.show_numbers(allr["id"]), "the same question"
    assert sub["predictions_reused"] and sub["results"] == allr["results"] and sub["id"] != allr["id"] and sub["n_frames"] == 6, "a subset costs no network"
    # Runner.predict + score = Evaluation.run
    pr = ev.runner.predict(net, "val")
    assert pr["results"] == allr["results"] and pr["frames_predicted"] == 0 and ev.score(pr["results"], "val")["id"] == allr["id"]
    _raises(EvaluationError, "not in", ev.score, pr["results"], "scenes:demo-9")
    # a new ground truth: another evaluation id, scored from the stored predictions, no network and no input loaded
    STATE.gt, STATE.shift, STATE.calls, STATE.loads = "g2", 3, 0, 0
    r3 = ev.run(net, "val", gt_key="g2", progress=0)
    assert STATE.calls == 0 and STATE.loads == 0 and r3["predictions_reused"] and r3["results"] == allr["results"] and r3["id"] != allr["id"], "rescored without the network"
    assert ev.show_numbers(r3["id"])["numbers"]["iou.iou"] != ev.show_numbers(allr["id"])["numbers"]["iou.iou"], "the other ground truth gives other numbers"
    assert r3["warnings"] == [] and "loss.total" not in ev.show_numbers(r3["id"])["numbers"], "the predictions carry no loss: nothing computed against an earlier ground truth"
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
    c = ev.compare(allr["id"], low["id"])
    assert c["metrics"]["iou.iou"]["diff"] == ev.show_numbers(low["id"])["numbers"]["iou.iou"] - ev.show_numbers(allr["id"])["numbers"]["iou.iou"], "compare is variant - control"
    assert ev.compare(allr["id"], low["id"])["metrics"]["iou.iou"]["diff"] == c["metrics"]["iou.iou"]["diff"]
    # a run killed half-way resumes from the start of its scene, so the memory is right: the numbers are the engine's
    net3, STATE.calls, STATE.fail_at = "runs/thr0.3", 0, 9
    _raises(RuntimeError, "killed", ev.run, net3, "val", progress=0)
    stored = [x for x in ev.list(net3) if x["kind"] == "predictions"]
    assert len(stored) == 1 and stored[0]["frames"] == 8, stored
    STATE.fail_at = None
    done = ev.run(net3, "val", progress=0)
    _same(ev.show_numbers(done["id"])["numbers"], _engine(net3, Frames.parse("val")), "resumed run")
    # a lock held by a live writer refuses the second one with a message
    net4 = "runs/thr0.7"
    rid = ev.check(net4, "val")["results"].partition("@")[2]
    os.makedirs(os.path.join(out(net4), "results"), exist_ok=True)
    if results_mod.fcntl is not None:
        with writer_lock(os.path.join(out(net4), "results", rid)):
            _raises(EvaluationError, "being written by pid", ev.run, net4, "val", progress=0)
        assert ev.run(net4, "val", progress=0)["id"], "the lock is free again"
    # the identity of the predictions: a checkpoint that changed, or another control, is another ResultSet (never the stored frames of the old one)
    base = ev.runner.predict(net, "scenes:demo-0:val")["results"]
    STATE.bump = 1
    try:
        moved = ev.runner.predict(net, "scenes:demo-0:val")
        assert moved["results"] != base and moved["frames_predicted"] == 6, "another checkpoint: predicted again under another id"
    finally:
        STATE.bump = 0
    ctl = ev.runner.predict(net, "scenes:demo-0:val", control="camera_off")
    assert ctl["results"] not in (base, moved["results"]) and ev.check(net, "scenes:demo-0:val", control="camera_off")["results"] == ctl["results"], "another control: another set"
    # the same frames asked through another set: another evaluation id (never a clash in one store), a clear message otherwise
    one, two = ev.run(net, "scenes:demo-0:val", progress=0), ev.run(net, "scenes:demo-0:small", progress=0)
    assert one["id"] != two["id"] and one["results"] == two["results"], "the set is part of the question; the predictions are shared"
    # the evaluator code of two evaluations differs: compare says so
    STATE.ev = "y"
    try:
        low_y = ev.run(net, "val", progress=0)
    finally:
        STATE.ev = "x"
    assert low_y["id"] != allr["id"] and low_y["results"] == allr["results"], "another evaluator hash: scored again from the same predictions"
    assert any("different evaluator code" in w for w in ev.compare(allr["id"], low_y["id"]).get("warnings", [])), "compare names a difference of the evaluator"
    # the evaluation of an id that is being written is refused with the holder's name
    if results_mod.fcntl is not None:
        eid = ev.check(net, "val", metrics=("maps",))["id"].partition("@")[2]
        with writer_lock(os.path.join(out(net), "suite", "quant", eid)):
            _raises(EvaluationError, "being written by pid", ev.score, allr["results"], "val", None, ("maps",))
        assert ev.score(allr["results"], "val", None, ("maps",))["id"].endswith(eid)
    # an adapter without the label-free pair: the Runner reads the labelled source and subject (more data than it needs, the same numbers)
    plain_ad = SimpleNamespace(**{f: getattr(ME, f) for f in ADAPTER_FUNCTIONS}, reference=reference, card=card)
    fb = Evaluation(adapter=plain_ad).run("runs/thr0.45", "val", progress=0)
    _same(Evaluation(adapter=plain_ad).show_numbers(fb["id"])["numbers"], _engine("runs/thr0.45", Frames.parse("val")), "the fallback path")
    # the adapter: free_source and free_subject go together
    half = SimpleNamespace(**{f: getattr(ME, f) for f in ADAPTER_FUNCTIONS}, free_source=free_source)
    _raises(EvaluationError, "go together", load_adapter, half, None)
    _raises(EvaluationError, "go together", Evaluation, None, half)
    # the identity of the network changes while it is loaded (a training wrote a checkpoint): nothing is stored, the message says why
    STATE.change = True
    try:
        _raises(EvaluationError, "changed while it was loaded", ev.runner.predict, "runs/thr0.8", "scenes:demo-0:val")
    finally:
        STATE.change, STATE.bump = False, 0
    assert all(x["frames"] == 0 for x in ev.list("runs/thr0.8")), "no frame was stored"
    # intermediate results: asked for with the predictions; a frame stored without them is predicted again, one with them is reused; unknown taps are refused before the network runs
    net5, STATE.calls = "runs/thr0.6", 0
    plain = ev.runner.predict(net5, "scenes:demo-0:val")
    assert plain["frames_predicted"] == 6
    with_tap = ev.runner.predict(net5, "scenes:demo-0:val", taps=("feat",))
    assert with_tap["frames_predicted"] == 6 and with_tap["results"] == plain["results"], "stored without the tap: predicted again to get it"
    STATE.calls = 0
    assert ev.runner.predict(net5, "scenes:demo-0:val", taps=("feat",))["frames_predicted"] == 0 and STATE.calls == 0 and ev.runner.predict(net5, "scenes:demo-0:val")["frames_predicted"] == 0, "and reused"
    assert ev.runner.tap(plain["results"], "demo-0/2")["taps"] == ["feat"]
    t = ev.runner.tap(plain["results"], "demo-0/2", "feat")
    assert t["shape"] == [32, 32] and np.load(t["path"]).shape == (32, 32)
    _raises(EvaluationError, "has no tap", ev.runner.tap, plain["results"], "demo-0/2", "memory_conf")
    _raises(EvaluationError, "unknown taps", ev.runner.predict, net5, "val", taps=("nothing",))
    assert ev.runner.predict(net5, "scenes:demo-0:val", taps="feat")["frames_predicted"] == 0 and ev.runner.plan(net5, "scenes:demo-0:val", taps="feat")["frames_stored"] == 6, "a bare string is one tap, not its letters"
    assert ev.check(net5, "scenes:demo-0:val", taps=("feat",))["frames_stored"] == 6 and ev.check(net5, "scenes:demo-0:val", taps=("memory_conf",))["frames_stored"] == 0, "check asks for taps like run"
    # the health of the memory (D70): healthy = no warning; full from the start, or never filling = a warning in the predictions' result
    assert plain["warnings"] == [] and ev.run(net5, "scenes:demo-0:val", progress=0)["warnings"] == []
    for mode, text in (("stuck", "not empty at the first frame"), ("flat", "does not fill")):
        STATE.memory = mode
        r = ev.run(f"runs/thr0.{4 if mode == 'stuck' else 2}", "scenes:demo-1:val", progress=0)
        assert len(r["warnings"]) == 1 and text in r["warnings"][0] and "demo-1" in r["warnings"][0], (mode, r["warnings"])
    STATE.memory = "ok"
    # the health at its boundaries (D70): 3 frames are too few to judge the filling; a start of exactly 0.05 and a growth of exactly 0.02 are healthy
    def health(fill):
        rs_ = SimpleNamespace(manifest={})
        Runner._memory_health(rs_, "s", fill)
        return rs_.manifest["health"]
    assert health([0.0, 0.0, 0.0]) == {} and health([0.05, 0.1, 0.2, 0.3]) == {} and health([0.0, 0.02, 0.0, 0.0]) == {} and health([]) == {}
    assert "not empty" in health([0.06, 0.2, 0.4, 0.6])["s"] and "does not fill" in health([0.0, 0.01, 0.0, 0.01])["s"]
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
    assert type(copy.deepcopy(allr)) is dict and pickle.loads(pickle.dumps(allr)) == allr, "the result of run is plain data"
    return ev, net, allr, low


def check_reading(ev, net, allr, low):
    q = ev.show_numbers(allr["id"], "iou")
    assert q["n_frames"] == 18 and set(q["numbers"]) >= {"iou.iou"} and q["better"]["iou.iou"] == "high" and all("iou" in k for k in q["numbers"])
    assert set(ev.split_numbers(allr["id"], "scene")) == {"demo-0", "demo-1", "demo-2"} and len(ev.split_numbers(allr["id"], "frame")) == 18
    _raises(EvaluationError, "axis", ev.split_numbers, allr["id"], "range")
    assert set(ev.find_worst(allr["id"], n=2)) == {"scenes", "frames", "flags"}
    assert ev.get_cost(allr["id"])["predict"]["n"] == 18 and ev.get_cost(allr["id"])["devices"] == ["cuda"]
    assert os.path.isfile(ev.report(allr["id"])) and ev.make_card(allr["id"])["name"] == allr["id"]
    rows = ev.list(net)
    assert {r["kind"] for r in rows} == {"predictions", "evaluation"} and allr["id"] in {r["id"] for r in rows} and allr["results"] in {r["id"] for r in rows}
    assert all(r["megabytes"] >= 0 for r in rows if r["kind"] == "predictions")
    assert ev.list_sets() == {"val": 3, "small": 1}
    _raises(EvaluationError, "not an evaluation id", ev.show_numbers, allr["results"])
    _raises(EvaluationError, "no evaluation", ev.show_numbers, "thr0.5@edeadbeef0000")
    assert Evaluation(adapter=ME).show_numbers(allr["id"])["numbers"] == ev.show_numbers(allr["id"])["numbers"], "a new object finds the evaluation by its id"
    pic = ev.draw_frame(allr["id"], "demo-1/4")
    assert os.path.isfile(pic["images"]["overview"]) and "iou.iou" in pic["scores"]
    gone = os.path.join(out(net), "results", allr["results"].partition("@")[2], "manifest.json")
    os.rename(gone, gone + ".away")                                                     # the predictions are gone (package.py clean keeps the manifest, a person may not): a card or a picture cannot be certified
    try:
        _raises(EvaluationError, "are gone", ev.make_card, allr["id"])
        _raises(EvaluationError, "are gone", ev.draw_frame, allr["id"], "demo-1/4")
    finally:
        os.rename(gone + ".away", gone)


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
    code, text, _ = call("Evaluation.show_numbers", info["id"], "--match", "iou", "--json", *base)
    assert code == 0 and json.loads(text)["numbers"]["iou.iou"] == Evaluation(adapter=ME).show_numbers(info["id"])["numbers"]["iou.iou"]
    assert call("quant", info["id"], *base)[0] is None and call("probe", "x", *base)[0] is None and call("list", "x", *base)[0] is None, "older command words are not taken over"
    code, text, _ = call("check", "runs/thr0.5", "val", *base)
    assert code == 0 and "ok: True" in text and "frames_stored: 18" in text, text
    code, text, _ = call("draw-frame", info["id"], "demo-0/2", *base)                         # the hyphen form of draw_frame
    assert code == 0 and ".png" in text, text
    code, text, _ = call("methods")
    assert code == 0 and "Evaluation.run(" in text and "Evaluation.compare(" in text and "Evaluation.tap(" in text and "gt_key=None" in text and not re.search(r"Runner\.\w+\(", text), "the Evaluation is the one front door of the command line"
    code, _, err = call("Evaluation.show_numbers", "thr0.5@e0000", *base)
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
    # the layers: the command line offers the Evaluation only; what executes the network (tap, time, probe) is passed to the Runner
    assert call("Runner.run", "runs/thr0.5", "val", *base)[0] is None, "the Runner is not on this command line (runsuite has it)"
    code, text, _ = call("tap", info["results"], "demo-0/2", *base)
    assert code == 0 and "taps" in text, text
    for word, what in (("time", "no timing"), ("probe", "no probe")):
        code, _, err = call("Evaluation." + word, "runs/thr0.5", *base)
        assert code == 2 and what in err, (word, code, err)
    code, text, _ = call("Evaluation.list", "runs/thr0.5", *base)
    assert code == 0 and "predictions" in text


def run_all():
    check_frames()
    check_codec()
    check_results()
    ev, net, allr, low = check_run()
    check_reading(ev, net, allr, low)
    check_cli()
    shutil.rmtree(STATE.root, ignore_errors=True)
    return {"ok": True, "checks": 5, "what": "frames, result set and locks, run = engine on the kept frames (warm), reuse / subset / rescoring / resume / predict+score / check, reading methods, command line"}


if __name__ == "__main__":
    print(json.dumps(run_all()))

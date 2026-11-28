"""Known answers of the class interface (api.py, netrunner.py, results.py) on a toy project: the discs of the minimal adapter, a "network" with a memory (its threshold falls with the
frames it has seen since the scene started); its Runner side (free_source / free_subject) reads no ground truth. This module is its own adapter. No data, no GPU, seconds.

What it proves: Frames round-trips and resolves; a ResultSet returns exactly what was stored and belongs to an identity, two writers exclude each other, a dead writer's lock is taken over;
`run` equals the engine's own run on the kept frames, also for one frame in the middle of a scene (the scene runs whole and warm: D70); a second `run`, and a subset of an earlier one, take
the stored predictions; a new ground truth is scored from them without the network; a run that was killed half-way resumes with the right memory; `check` finds the problems before
anything runs; the reading methods and the pictures work from what is stored; the command line runs them."""
from __future__ import annotations

import ast
import contextlib
import copy
import functools
import io
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np

from .. import cli, defaults, minimal
from .. import results as results_mod
from ..api import Evaluation, EvaluationError, Frames, load_adapter
from ..contracts import FrameRef, Prediction, Sample, TapSpec
from ..netrunner import Runner
from ..render import figure
from ..results import Locked, ResultSet, decode, encode, result_id, writer_lock
from ..suite import EvalSuite
from . import check, raises

# ------------------------------------------------------------------------------------------------ the toy project (this module is its own adapter)
STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False, ev="x")
MEMORY_TAP = "memory_conf"
COMPARE_IGNORE = {"source": ("evaluator_hash",)}                                   # the arms may differ in the evaluator code: compare says so in its warnings
METRICS = ("maps", "boxes")                     # what an evaluation uses when it names none


def name(network): return os.path.basename(network)
def network(nm): return nm
def out(net): return os.path.join(STATE.root, name(net))
def sets(): return {"val": 3, "small": 1}
def identity(net): return {"network": net, "checkpoint": STATE.bump}
def evaluator_now(): return STATE.ev                                                # the evaluator code as it is now (what show_numbers compares a stored evaluation with)


class Source(minimal.Discs):
    def __init__(self, of):
        super().__init__(of)

    def scenes(self):
        return ["demo-0"] if self.split == "small" else super().scenes()

    def load(self, ref):
        STATE.loads += 1
        return super().load(ref)

    def describe(self):
        return {"class": "ToySource", "dataset": "toy", "split": self.split, "scenes": len(self.scenes()), "gt_signature": STATE.gt, "gt_version": None, "evaluator_hash": STATE.ev}


class Subject(minimal.Threshold):
    """a network with a memory: the threshold falls by 0.04 for every frame seen since reset()"""
    def __init__(self, threshold):
        super().__init__(threshold)
        self.seen = 0

    def reset(self):
        self.seen = 0

    def taps(self):
        return [TapSpec("memory_conf", "map", "memory", "the fill of the memory", extent=(0, 32, 0, 32), flip_x=True, alive_check=False),      # empty at a scene start
                TapSpec("feat", "map", "input", "the input picture"), TapSpec("dead", "map", "input", "always zero")]

    def predict(self, sample, taps=()):
        STATE.calls += 1
        if STATE.fail_at and STATE.calls >= STATE.fail_at:
            raise RuntimeError("killed")
        thr = self.threshold - 0.04 * self.seen
        fill = {"ok": 0.1 * self.seen, "stuck": 0.3, "flat": 0.0}[STATE.memory]      # empty at the start of a scene and growing / full from the start / never filling
        self.seen += 1
        tapped = {"memory_conf": np.full((2, 2), fill), "feat": np.asarray(sample.inputs, np.float32), "dead": np.zeros((2, 2))}
        return Prediction(outputs=sample.inputs > thr, taps={k: v for k, v in tapped.items() if k in taps}, info={"predict_ms": 1.0})

    def describe(self):
        return {"class": "ToySubject", "threshold": self.threshold}


class FreeSource(Source):
    """the data the Runner reads: no ground truth (reference None), its own load counter; it describes what the network is fed like the labelled source does"""
    def __init__(self, of, control=""):
        super().__init__(of)
        self.control = control

    def load(self, ref):
        STATE.free_loads += 1
        s = minimal.Discs.load(self, ref)
        return Sample(s.ref, inputs=s.inputs, reference=None, meta={"token": ref.key})

    def describe(self):
        return {"class": "FreeSource", "dataset": "toy", "split": self.split, "scenes": len(self.scenes()), "control": self.control or None}


def source(net, of, workers=0, control=""): return Source(of)                       # with ground truth: what the Evaluation scores with
def free_source(net, of, workers=0, control=""): return FreeSource(of, control)       # without: what the Runner runs on
def subject(net, src, device): return Subject(float(net.split("thr")[-1]))


def free_subject(net, src, device):                                                  # (the toy network never reads a ground truth)
    STATE.bump += 1 if STATE.change else 0                                           # "a training wrote a new checkpoint while the network was loaded"
    return subject(net, src, device)


def task(net, metrics):
    if set(metrics) - {"maps", "boxes"}:
        raise ValueError(f"unknown metrics {sorted(set(metrics) - {'maps', 'boxes'})}")
    return minimal.DiscTask()


def reference(src, ref):
    truth = np.asarray(minimal.Discs.load(src, ref).reference)
    return np.roll(truth, STATE.shift, axis=1) if STATE.shift else truth          # a "new ground truth": the disc moved


ME = sys.modules[__name__]
FUNCTIONS = ("name", "out", "network", "sets", "source", "subject", "task", "identity")


def toy_eval(adapter=None) -> Evaluation:
    return Evaluation(adapter=adapter or ME)


def fresh() -> None:
    STATE.__dict__.update(root=tempfile.mkdtemp(), gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False, ev="x")


# ------------------------------------------------------------------------------------------------ the checks
def _engine(net, frames_spec):
    """the numbers of the engine's own run: the scenes whole, state carry, scored on the kept frames only"""
    src, subj = minimal.Discs(), Subject(float(net.split("thr")[-1]))
    suite = EvalSuite(src, subj, minimal.DiscTask(), out=tempfile.mkdtemp(), state="carry")
    keep = frames_spec.resolve(src)[1]
    store = suite.quant.run(scenes=sorted({r.scene for r in keep}), name="eng", progress=0)
    return suite.quant.table(store, [r.label for r in keep])


def _same(got, want, what):
    assert set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-12 for k in want), (what, got, want)


@check("api: a frame text round-trips and resolves to the scenes to run and the frames to keep; a bad one is a message")
def _():
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
        raises(EvaluationError, "", Frames.parse, bad)
    raises(EvaluationError, "not in val", P("scenes:demo-9").resolve, src)
    raises(EvaluationError, "not a frame", P("frames:demo-0/99").resolve, src)
    raises(EvaluationError, "not a frame", P("frames:junk").resolve, src)
    raises(EvaluationError, "takes", Frames, "spread", ())


@check("results: the codec gives everything a prediction holds back with its types; nothing that needs pickle is stored")
def _():
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
    raises(TypeError, "object array", encode, {"a": np.array([None, 1], dtype=object)})
    raises(TypeError, "cannot store", encode, {"a": {1, 2}})
    raises(TypeError, "cannot store", encode, {"a": object()})


@check("results: a ResultSet belongs to its identity, keeps frames and taps, refuses a changed data set and a damaged file; one writer per folder, a dead writer's lock is free")
def _():
    tmp = tempfile.mkdtemp()
    refs = minimal.Discs().frames("demo-1")[:3]
    ident = {"network": {"name": "a"}, "inputs": {"x": 1}}
    rid = result_id(ident)
    assert rid == result_id({"inputs": {"x": 1}, "network": {"name": "a"}}) and rid != result_id({**ident, "inputs": {"x": 2}}), "the id is the identity, whatever the key order"
    raises(ValueError, "named by its id", ResultSet.create, os.path.join(tmp, "wrong"), ident)
    rs = ResultSet.create(os.path.join(tmp, rid), ident)
    assert rs.refs() == [] and not rs.has(refs[0])
    pred = Prediction(outputs={"dt": {0.0: {"prob": {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, "dets": [{"x": 1.5, "c": "car"}], "none": None}}},
                      taps={"t": np.ones(2)}, info={"predict_ms": 2.5})
    rs.add(refs[0], pred)
    back = rs.load(refs[0])
    assert np.array_equal(back.outputs["dt"][0.0]["prob"]["a"], pred.outputs["dt"][0.0]["prob"]["a"]) and back.outputs["dt"][0.0]["dets"] == [{"x": 1.5, "c": "car"}]
    assert back.outputs["dt"][0.0]["none"] is None and set(back.outputs["dt"]) == {0.0} and back.info == {"predict_ms": 2.5} and back.taps == {} and np.array_equal(rs.load(refs[0], ("t",)).taps["t"], np.ones(2)), "keys and types survive; taps come on request"
    assert rs.has(refs[0]) and not rs.has(refs[1])
    raises(KeyError, "not in result set", rs.load, refs[1])
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
    raises(IOError, "the dataset changed", rk.load, FrameRef("demo-9", 0, "other"))
    old = json.load(open(os.path.join(tmp, rid, "manifest.json")))
    old["format"] = 0
    os.makedirs(os.path.join(tmp, "oldset"))
    json.dump(old, open(os.path.join(tmp, "oldset", "manifest.json"), "w"))
    raises(ValueError, "result set format 0", ResultSet.open, os.path.join(tmp, "oldset"))
    raises(KeyError, "with the taps", rw.load, refs[2], ("feat",))
    raises(IOError, "read-only", again.add, refs[0], pred)
    assert not [f for f in os.listdir(os.path.join(tmp, rid, "frames")) if ".tmp" in f], "no half-written file stays"
    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.npz"), "wb") as f:
        f.write(b"not a frame")
    raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[0])
    good = open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "rb").read()
    mid = bytearray(good)
    with zipfile.ZipFile(io.BytesIO(good)) as z:
        off = z.getinfo("tree.json").header_offset
    nlen, elen = struct.unpack("<HH", good[off + 26: off + 30])
    mid[off + 30 + nlen + elen + 1] ^= 0xFF                                             # a byte inside the compressed json: a zlib / CRC error, not a traceback
    with open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "wb") as f:
        f.write(bytes(mid))
    raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[1])
    with open(os.path.join(tmp, rid, "frames", "notes.npz"), "wb") as f:                # a file that is not named <scene>_<index>.npz is not a frame
        f.write(b"x")
    listed = ResultSet.open(os.path.join(tmp, rid))
    assert {r.label for r in listed.refs()} == {r.label for r in refs} | {"demo-9/000"} and listed.n_frames() == 4, "notes.npz is not a frame"
    # a clean-up that removes the frames / taps folders and keeps the manifest: the set takes frames again
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
        raises(Locked, "being written by pid", writer_lock(target).__enter__)
    with writer_lock(target):                                                   # released: the next writer gets it; the lock file stays, its content is only the message
        pass
    with open(target + ".lock", "w") as f:                                      # a lock file left by a dead writer does not block
        f.write("pid 4194301 on another-host")
    with writer_lock(target):
        pass
    code = ("import sys; sys.path.insert(0, %r); from Evaluation.results import writer_lock; import time\n"
            "with writer_lock(%r):\n    print('held', flush=True); time.sleep(30)\n") % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), target)
    other = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)   # a second PROCESS holds it: this one is refused with the holder's pid
    try:
        assert other.stdout.readline().strip() == "held"
        raises(Locked, f"pid {other.pid}", writer_lock(target).__enter__)
    finally:
        other.kill()
        other.wait()
    with writer_lock(target):                                                   # killed holder: free at once
        pass


@check("api run: the engine's numbers on the kept frames (warm); stored predictions are reused, subsets, a new ground truth and a killed run; identity, taps, memory health, the dry run")
def _():
    fresh()
    ev = toy_eval()
    net = "runs/thr0.5"
    # the engine's own run is the reference: same numbers on the kept frames (the kept frame of one scene in the middle runs with the scene's history)
    for spec in ("val", "frames:demo-1/4", "scenes:demo-2:val", "spread:7"):
        r = ev.run(net, spec, gt_key="g1", progress=0)
        _same(ev.show_numbers(r["id"])["numbers"], _engine(net, Frames.parse(spec)), spec)
        assert r["n_frames"] == len(Frames.parse(spec).resolve(Source("val"))[1]) and r["warnings"] == []
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
    raises(EvaluationError, "not in", ev.score, pr["results"], "scenes:demo-9")
    # a new ground truth: another evaluation id, scored from the stored predictions, no network and no input loaded
    STATE.gt, STATE.shift, STATE.calls, STATE.loads = "g2", 3, 0, 0
    r3 = ev.run(net, "val", gt_key="g2", progress=0)
    assert STATE.calls == 0 and STATE.loads == 0 and r3["predictions_reused"] and r3["results"] == allr["results"] and r3["id"] != allr["id"], "rescored without the network"
    assert ev.show_numbers(r3["id"])["numbers"]["iou.iou"] != ev.show_numbers(allr["id"])["numbers"]["iou.iou"], "the other ground truth gives other numbers"
    assert r3["warnings"] == []
    STATE.gt, STATE.shift = "g1", 0
    raises(EvaluationError, "has ground truth g1", ev.run, net, "val", gt_key="g9")
    STATE.gt = None
    raises(EvaluationError, "no ground-truth signature", ev.run, net, "val")
    STATE.gt = "g1"
    raises(EvaluationError, "not in val", ev.run, net, "scenes:demo-7")
    raises(EvaluationError, "unknown metrics", ev.run, net, "val", metrics=("mapz",))
    # another network = other predictions and another id; the sign of compare is variant - control
    low = ev.run("runs/thr0.1", "val", progress=0)
    assert low["results"] != allr["results"] and low["id"] != allr["id"]
    c = ev.compare(allr["id"], low["id"])
    assert c["metrics"]["iou.iou"]["diff"] == ev.show_numbers(low["id"])["numbers"]["iou.iou"] - ev.show_numbers(allr["id"])["numbers"]["iou.iou"], "compare is variant - control"
    # a run killed half-way resumes from the start of its scene, so the memory is right: the numbers are the engine's
    net3, STATE.calls, STATE.fail_at = "runs/thr0.3", 0, 9
    raises(RuntimeError, "killed", ev.run, net3, "val", progress=0)
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
            raises(EvaluationError, "being written by pid", ev.run, net4, "val", progress=0)
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
    assert not any("scored with evaluator code" in w for w in ev.show_numbers(allr["id"])["warnings"])
    STATE.ev = "z"                                                                      # the evaluator code changed after allr was scored
    try:
        assert any("scored with evaluator code" in w for w in ev.show_numbers(allr["id"])["warnings"]), "show_numbers says the counts are reduced with today's metrics"
    finally:
        STATE.ev = "x"
    sub_c = ev.compare(allr["id"], sub["id"])
    assert sub_c["n_scenes"] == 1 and sub_c["only_a"] == 12 and sub_c["n_common_frames"] == 6, sub_c
    assert any("share 6 frames only" in w for w in sub_c["warnings"]) and any("only 1 scenes" in w for w in sub_c["warnings"]), "an interval on a subset and on few scenes says so"
    # an adapter that gives no evaluator hash (a file that shapes the numbers is missing or renamed) fails closed: nothing is reused across metric edits that nobody can see
    STATE.ev = None
    try:
        raises(EvaluationError, "no evaluator hash", ev.run, net, "val", progress=0)
        chk_ev = ev.check(net, "val")
        assert not chk_ev["ok"] and any("no evaluator hash" in p for p in chk_ev["problems"]), chk_ev
        assert any("cannot be hashed now" in w for w in ev.show_numbers(allr["id"])["warnings"]), "an evaluation made earlier says its code can no longer be compared"
    finally:
        STATE.ev = "x"
    # the evaluation of an id that is being written is refused with the holder's name
    if results_mod.fcntl is not None:
        eid = ev.check(net, "val", metrics=("maps",))["id"].partition("@")[2]
        with writer_lock(os.path.join(out(net), "suite", "quant", eid)):
            raises(EvaluationError, "being written by pid", ev.score, allr["results"], "val", None, ("maps",))
        assert ev.score(allr["results"], "val", None, ("maps",))["id"].endswith(eid)
    # an adapter without the label-free pair: the Runner reads the labelled source and subject (more data than it needs, the same numbers)
    plain_ad = SimpleNamespace(**{f: getattr(ME, f) for f in FUNCTIONS}, METRICS=METRICS, reference=reference)
    fb = toy_eval(plain_ad).run("runs/thr0.45", "val", progress=0)
    _same(toy_eval(plain_ad).show_numbers(fb["id"])["numbers"], _engine("runs/thr0.45", Frames.parse("val")), "the fallback path")
    # the adapter: free_source and free_subject go together, and its required functions
    half = SimpleNamespace(**{f: getattr(ME, f) for f in FUNCTIONS}, METRICS=METRICS, free_source=free_source)
    raises(EvaluationError, "go together", load_adapter, half, None)
    raises(EvaluationError, "go together", Evaluation, None, half)
    raises(EvaluationError, "lacks source, subject, task", load_adapter, SimpleNamespace(sets=sets), None)
    # the identity of the network changes while it is loaded (a training wrote a checkpoint): nothing is stored, the message says why
    STATE.change = True
    try:
        raises(EvaluationError, "changed while it was loaded", ev.runner.predict, "runs/thr0.8", "scenes:demo-0:val")
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
    assert ev.tap(plain["results"], "demo-0/2")["taps"] == ["feat"]
    t = ev.tap(plain["results"], "demo-0/2", "feat")
    assert t["shape"] == [32, 32] and np.load(t["path"]).shape == (32, 32)
    raises(EvaluationError, "has no tap", ev.tap, plain["results"], "demo-0/2", "memory_conf")
    raises(EvaluationError, "unknown taps", ev.runner.predict, net5, "val", taps=("nothing",))
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
    raises(EvaluationError, "not one of", ev.run, net, "nosuch")
    assert ev.check("runs/thr0.9", "spread:3")["network_runs"] is True
    bad = ev.check(net, "scenes:demo-9", metrics=("mapz",))
    assert not bad["ok"] and bad["problems"], bad
    assert not ev.check(net, "frames:demo-1/4", metrics=("mapz",))["ok"]
    STATE.gt = None
    assert not ev.check(net, "val")["ok"]
    STATE.gt = "g1"
    assert type(copy.deepcopy(allr)) is dict and pickle.loads(pickle.dumps(allr)) == allr, "the result of run is plain data"
    shutil.rmtree(STATE.root, ignore_errors=True)


@check("api adapter: a \"runner\" key in evaluation.json and an adapter function `inputs` are refused, not silently ignored; an evaluation.json that is no json object is an error "
       "with the file and what is wrong, a missing one is not")
def _():
    project = tempfile.mkdtemp()
    os.makedirs(os.path.join(project, "evaluation"))
    cfg = os.path.join(project, "evaluation", "evaluation.json")

    def config(**keys):
        with open(cfg, "w") as f:
            json.dump({"adapter": __name__, **keys}, f)
    config(runner="mine:Runner")
    for adapter in (None, ME):                                         # also when the adapter is passed in
        raises(EvaluationError, '"runner" is not supported', Evaluation, project, adapter)
    config()
    assert Evaluation(project).ad.sets is sets
    for text, problem in (('{"adapter": ', "cannot be read: JSONDecodeError: Expecting value: line 1 column 13"), ('["' + __name__ + '"]', "must hold a json object")):
        with open(cfg, "w") as f:
            f.write(text)
        for adapter in (None, ME):                                     # the file is the project's: broken is broken whoever names the adapter
            raises(EvaluationError, f"{cfg} {problem}", Evaluation, project, adapter)
    os.remove(cfg)
    raises(EvaluationError, "no adapter: pass one", Evaluation, project)
    assert Evaluation(project, ME).ad.sets is sets, "no evaluation.json and an adapter passed in"
    hooked = SimpleNamespace(**{f: getattr(ME, f) for f in FUNCTIONS}, inputs=lambda source: {})
    raises(EvaluationError, "`inputs`, which is not supported", Evaluation, None, hooked)
    shutil.rmtree(project, ignore_errors=True)


@check("holdout: a split of problem.json's `holdout` is refused by run, check and guard before anything is read: without a gate, with the variables of a TEST run set by hand, "
       "and a gate of the caller's own is not taken; the other splits stay open")
def _():
    fresh()
    project = tempfile.mkdtemp()
    with open(os.path.join(project, "problem.json"), "w") as f:
        json.dump({"holdout": ["test"]}, f)
    held = SimpleNamespace(**{f: getattr(ME, f) for f in FUNCTIONS}, METRICS=METRICS, reference=reference)
    held.sets = lambda: {"val": 3, "test": 3, "small.test": 1}
    ev = Evaluation(project, held)
    variables = ("RESEARCH_RUN", "RESEARCH_HOLDOUT_RUN", "RESEARCH_PROBLEM")
    saved, gate = {k: os.environ.pop(k, None) for k in variables}, defaults._holdout_gate

    def closed(why):
        for frames in ("test", "small.test", "scenes:demo-0:test"):
            raises(EvaluationError, "is a holdout split", ev.run, "runs/thr0.5", frames, progress=0)
        chk = ev.check("runs/thr0.5", "test")
        assert not chk["ok"] and "is a holdout split" in chk["problems"][0], (why, chk)
        raises(PermissionError, "is a holdout split", defaults.guard, "test", project)
        assert not defaults.holdout_open() and STATE.calls == STATE.loads == STATE.free_loads == 0, why
    try:
        defaults._holdout_gate = None                                  # as in every process the research core did not hand its gate to
        closed("no gate, no TEST run")
        os.environ.update(RESEARCH_RUN="E1.a_s0", RESEARCH_HOLDOUT_RUN="E1.a_s0", RESEARCH_PROBLEM="toy")
        closed("the variables of a TEST run set by hand")
        def test_refusal(problem, run):                                # a yes of the caller's own, dressed as the gate of the research core
            return ""
        test_refusal.__module__ = "research.gates"

        class Yes:
            __module__ = "research.gates"

            def __init__(self):
                self.__name__ = "test_refusal"

            def __call__(self, problem, run):
                return ""
        for own in (lambda problem, run: "", test_refusal, functools.wraps(test_refusal)(lambda problem, run: ""), Yes()):
            raises(PermissionError, "the holdout gate is research.gates.test_refusal", defaults.set_holdout_gate, own)
            closed(f"a gate of the caller's own: {own!r}")
        assert ev.run("runs/thr0.5", "val", progress=0)["n_frames"] == 18 and defaults.guard("val", project) is None, "a split that is no holdout is open"
    finally:
        defaults._holdout_gate = gate
        for k in variables:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
    shutil.rmtree(project, ignore_errors=True)
    shutil.rmtree(STATE.root, ignore_errors=True)


@check("imports: no module of the package imports the research core (by statement, by a named import_module, or when it is loaded), and none imports a module of the package "
       "inside a function (there is no import cycle to work around)")
def _():
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad, modules = [], []
    for folder, dirs, files in os.walk(package):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
        for name in (f for f in files if f.endswith(".py")):
            where = os.path.relpath(os.path.join(folder, name), package)
            if name not in ("__init__.py", "__main__.py", "selftest.py"):      # (selftest.py loads the tests)
                modules.append("Evaluation." + where[:-3].replace(os.sep, "."))
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            top = {id(n) for n in tree.body}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("import_module", "__import__"):
                    mods, own = [a.value for a in node.args[:1] if isinstance(a, ast.Constant) and isinstance(a.value, str)], False
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                    own = isinstance(node, ast.ImportFrom) and node.level > 0 or any(m.split(".")[0] == "Evaluation" for m in mods)
                else:
                    continue
                if any(m.split(".")[0] == "research" for m in mods) or (own and id(node) not in top):
                    bad.append(f"{where}:{node.lineno}")
    assert not bad and len(modules) > 15, (bad, modules)
    loaded = ("import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
              "print(sorted(m for m in sys.modules if m.split('.')[0] == 'research'))\n")
    r = subprocess.run([sys.executable, "-c", loaded] + modules, cwd=os.path.dirname(package), capture_output=True, text=True)      # from autoresearch/: `research` could be imported
    assert r.returncode == 0 and r.stdout.strip() == "[]", (r.stdout, r.stderr[-500:])


@check("api reading: numbers, splits, worst frames, cost, report, counts for the judge, listing - and a picture drawn from the stored prediction (no network), refused without it")
def _():
    fresh()
    ev = toy_eval()
    net = "runs/thr0.5"
    allr = ev.run(net, "val", progress=0)
    q = ev.show_numbers(allr["id"], "iou")
    assert q["n_frames"] == 18 and set(q["numbers"]) >= {"iou.iou"} and q["better"]["iou.iou"] == "high" and all("iou" in k for k in q["numbers"])
    assert set(ev.split_numbers(allr["id"], "scene")) == {"demo-0", "demo-1", "demo-2"} and len(ev.split_numbers(allr["id"], "frame")) == 18
    raises(EvaluationError, "axis", ev.split_numbers, allr["id"], "range")
    worst = ev.find_worst(allr["id"], n=2)
    assert set(worst) == {"scenes", "frames", "flags"} and worst["frames"][0]["view"] == f"draw-frame {allr['id']} {worst['frames'][0]['label']}"
    assert ev.get_cost(allr["id"])["predict"]["n"] == 18 and ev.get_cost(allr["id"])["devices"] == ["cuda"]
    assert os.path.isfile(ev.report(allr["id"]))
    rows = ev.list(net)
    assert {r["kind"] for r in rows} == {"predictions", "evaluation"} and allr["id"] in {r["id"] for r in rows} and allr["results"] in {r["id"] for r in rows}
    assert all(r["megabytes"] >= 0 for r in rows if r["kind"] == "predictions")
    assert ev.list_sets() == {"val": 3, "small": 1}
    metric, store = ev.metrics_of(allr["id"])["iou"], ev.stored_counts(allr["id"])
    assert metric.reduce([c for cs in store.by_scene("iou").values() for c in cs])["iou"] == ev.show_numbers(allr["id"])["numbers"]["iou.iou"] and "stored_counts" not in Evaluation.METHODS
    raises(EvaluationError, "not an evaluation id", ev.show_numbers, allr["results"])
    raises(EvaluationError, "no evaluation", ev.show_numbers, "thr0.5@edeadbeef0000")
    assert toy_eval().show_numbers(allr["id"])["numbers"] == ev.show_numbers(allr["id"])["numbers"], "a new object finds the evaluation by its id"
    pages, real = [], figure.render_page
    figure.render_page = lambda page, path=None, **kw: (pages.append(page), real(page, path, **kw))[1]          # what is drawn, next to the file
    try:
        STATE.calls = 0
        pic = ev.draw_frame(allr["id"], "demo-1/4")
        assert os.path.getsize(pic["image"]) > 1000 and pic["frame"] == "demo-1/004" and pic["scores"] == ev.split_numbers(allr["id"], "frame")["demo-1/004"], "the stored prediction, the stored scores"
        assert pages[-1].title == f"demo-1/004  [overview]   iou.iou {pic['scores']['iou.iou']:.3f}\ndisc", ("the label and the scores head the view's own title", pages[-1].title)
        target = os.path.join(STATE.root, "pictures", "one.png")
        assert ev.draw_frame(allr["id"], "demo-1/4", out=target)["image"] == target and os.path.isfile(target)
        raises(EvaluationError, "no view 'nothing'", ev.draw_frame, allr["id"], "demo-1/4", "nothing")
        raises(EvaluationError, "none are stored", ev.draw_frame, allr["id"], "demo-1/4", "debug")
        assert STATE.calls == 0, "no picture ran the network"
        ev.run(net, "frames:demo-1/0", taps=("memory_conf", "feat", "dead"), progress=0)
        STATE.calls = 0
        ev.draw_frame(allr["id"], "demo-1/0", "debug")                  # the inside of the network, from the stored taps, as the subject describes them
        page = pages[-1]
    finally:
        figure.render_page = real
    assert STATE.calls == 0 and page.title.startswith("demo-1/000  [debug]   iou.iou "), page.title
    memory = next(p for p in page.panels if p.title == "[memory] memory_conf")
    assert memory.extent == (0, 32, 0, 32) and memory.xlim == (32, 0) and memory.note == "the fill of the memory", "stage, plane, flip and note of the TapSpec"
    table = " ".join(str(row) for row in page.panels[-1].data)
    assert "dead: alive" in table and "memory_conf: alive" not in table, "the flags follow each TapSpec's alive check"
    STATE.gt = "g2"
    raises(EvaluationError, "would not show what was scored", ev.draw_frame, allr["id"], "demo-1/4")
    STATE.gt = "g1"
    part = ev.run("runs/thr0.55", "frames:demo-0/1", progress=0)
    raises(EvaluationError, "not among the predictions", ev.draw_frame, part["id"], "demo-2/0")
    gone = os.path.join(out(net), "results", allr["results"].partition("@")[2], "manifest.json")
    os.rename(gone, gone + ".away")                                                     # the predictions are gone: there is nothing to draw
    try:
        raises(EvaluationError, "are gone", ev.draw_frame, allr["id"], "demo-1/4")
    finally:
        os.rename(gone + ".away", gone)
    shutil.rmtree(STATE.root, ignore_errors=True)


@check("api on the command line: every method is a command on a project's adapter; json, exit codes and messages")
def _():
    fresh()
    base = ["--adapter", __name__, "--project", STATE.root]
    toy_eval().run("runs/thr0.5", "val", progress=0)

    def call(*argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, buf.getvalue(), err.getvalue()
    code, text, _ = call("Evaluation.run", "runs/thr0.5", "val", "--json", "--workers", "0", "--progress", "0", *base)
    info = json.loads(text)                                                                   # stdout is the json and nothing else
    assert code == 0 and info["frames"] == "val" and info["predictions_reused"] is True and info["id"].startswith("thr0.5@e"), (code, text[:200])
    code, text, _ = call("run", "runs/thr0.5", "--metrics", "maps", "--json", "--workers", "0", "--progress", "0", *base)
    assert code == 0 and json.loads(text)["metrics"] == ["maps"], text
    code, text, _ = call("show_numbers", info["id"], "--match", "iou", "--json", *base)
    assert code == 0 and json.loads(text)["numbers"]["iou.iou"] == toy_eval().show_numbers(info["id"])["numbers"]["iou.iou"]
    code, text, _ = call("check", "runs/thr0.5", "val", *base)
    assert code == 0 and "ok: True" in text and "frames_stored: 18" in text, text
    code, text, _ = call("draw-frame", info["id"], "demo-0/2", *base)                         # the hyphen form of draw_frame
    assert code == 0 and ".png" in text, text
    code, text, _ = call("methods")
    assert code == 0 and "Evaluation.run(" in text and "Evaluation.compare(" in text and "gt_key=None" in text and "stored_counts" not in text and "metrics_of" not in text
    code, _, err = call("show-numbers", "thr0.5@e0000", *base)
    assert code == 2 and "no evaluation" in err
    code, _, err = call("run", "runs/thr0.5", "scenes:demo-9", *base)                         # a data error is a message and exit 2, not a traceback
    assert code == 2 and "not in val" in err and "Traceback" not in err, err
    code, _, err = call("run", "runs/thr0.5", "val", "--metrics", "mapz", *base)
    assert code == 2 and "unknown metrics" in err, err
    code, text, _ = call("check", "runs/thr0.5", "scenes:demo-9", *base)
    assert code == 1 and "ok: False" in text, text
    code, text, _ = call("tap", info["results"], "demo-0/2", *base)
    assert code == 0 and "taps" in text, text
    for word, what in (("time", "no timing"), ("probe", "no probe")):
        code, _, err = call(word, "runs/thr0.5", *base)
        assert code == 2 and what in err, (word, code, err)
    code, text, _ = call("list", "runs/thr0.5", *base)
    assert code == 0 and "predictions" in text
    shutil.rmtree(STATE.root, ignore_errors=True)

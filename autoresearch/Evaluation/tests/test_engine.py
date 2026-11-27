"""Known answers of the engine: the data contracts, the run store, the statistics, the frame loop, the counting and the drawing."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np

from .. import EvalSuite, FrameRef, Page, Panel, Sample, Shape, TapSpec
from ..contracts import Flag
from ..render import figure, report, taps as tapmod
from ..runner import Runner
from ..sampling import Sampler
from ..stats import bootstrap_scenes, interval_draws, loo_influence, paired_draws, percentile_interval, robust_z, scene_draws, scene_picks
from ..store import CHUNK, RunStore, _identity_differences
from . import check, raises
from .toyworld import PLANTED, Smooth, ToyTask, Wrong, World


def make(out: str, seed: int = 0, bias: float = 0.0, state: str = "carry") -> EvalSuite:
    return EvalSuite(World(seed), Smooth(bias), ToyTask(), out=out, state=state)


def direct_iou(source, subject, scenes=None, reset: bool = False) -> float:
    """the iou computed without the engine: every frame through the subject, cells summed"""
    both = either = 0
    for scene in scenes or source.scenes():
        subject.reset()
        for ref in source.frames(scene):
            if reset:
                subject.reset()
            sample = source.load(ref)
            found = subject.predict(sample).outputs
            both, either = both + (found & sample.reference).sum(), either + (found | sample.reference).sum()
    return both / either


# ------------------------------------------------------------------------------------------------ data
@check("contracts: FrameRef labels and parsing (the key is no identity); lazy display values; Source.find by label, tuple, FrameRef and key, errors name the closest scenes")
def _():
    r = FrameRef("scene-0061", 12, "tok")
    assert r.label == "scene-0061/012" and r.slug == "scene-0061_012" and str(r) == r.label and r == FrameRef("scene-0061", 12, "other")
    assert FrameRef.parse("scene-0061/12") == ("scene-0061", 12) == FrameRef.parse("scene-0061:012") and FrameRef.parse("scene-0061") is None and FrameRef.parse("a/b") is None
    calls = []
    s = Sample(r, None, None, {"img": lambda: calls.append(1) or np.ones(2)})
    assert s.get("img").sum() == 2 and s.get("img").sum() == 2 and len(calls) == 1 and s.get("none", 5) == 5
    src = World()
    f = src.frames("toy-2")[7]
    assert src.find("toy-2/7") == f and src.find(("toy-2", 7)) == f and src.find(f) == f and src.find(f.key) == f
    for bad, text in (("toy-9/1", "closest"), ("toy-2/99", "frames 0..7"), ("nonsense", "not a frame key")):
        raises(KeyError, text, src.find, bad)


@check("sampling: random scenes and evenly spread frames are seeded and as documented; an unknown scene is refused")
def _():
    src = World()
    assert Sampler(src, 1).scenes(3) == Sampler(src, 1).scenes(3) and len(set(Sampler(src, 1).scenes(3))) == 3 and len(Sampler(src, 1).scenes(99)) == 4
    flat = [f for s in src.scenes() for f in src.frames(s)]
    assert [f.label for f in Sampler(src, 0).spread(10)] == [flat[i].label for i in sorted({int(i) for i in np.linspace(0, 31, 10).round()})]
    assert len(Sampler(src, 0).spread(1000)) == 32
    raises(KeyError, "unknown scene", Sampler(src, 0).spread, 5, ["no-such-scene"])


# ------------------------------------------------------------------------------------------------ the run store
@check("RunStore: round trip over several chunks (fixed and ragged counts), resume, a crash leaves a consistent prefix, other damage is refused and left as it is")
def _():
    with tempfile.TemporaryDirectory() as d:
        n = 2 * CHUNK + 17
        st = RunStore.create(os.path.join(d, "r"), {"name": "t"})
        rng = np.random.default_rng(0)
        want = []
        for i in range(n):
            c = {"fixed": rng.random((2, 3)), "ragged": rng.random(int(rng.integers(0, 6)))}
            want.append(c)
            st.add(FrameRef(f"s{i // 50}", i % 50, f"k{i}"), {"m": c}, {"m.a": float(i)}, {"m.a": 1.0}, [], {"ms": 1.0})
        st.close()
        r = RunStore.open(os.path.join(d, "r"))
        assert len(r.frames) == n and r.manifest["finished"] and r.manifest["n_frames"] == n
        for w, g in zip(want, r.counts("m")):
            assert np.array_equal(w["fixed"], g["fixed"]) and np.array_equal(w["ragged"], g["ragged"]) and g["ragged"].shape == w["ragged"].shape
        assert set(r.by_scene("m")) == {"s0", "s1", "s2"} and sum(len(v) for v in r.by_scene("m").values()) == n
        assert r.frame("s0/3") is r.frame("s0/003") is r.frame("s0:3") and r.frame("s0/003")["key"] == "k3" and len(r.counts("m", ["s0/003", "s1/002"])) == 2
        raises(KeyError, "no such frame", r.frame, "s9/0")
        assert RunStore.create(os.path.join(d, "r"), {"name": "t"}).done() == r.done()
        raises(FileExistsError, "already holds a run", RunStore.create, os.path.join(d, "r"), {"name": "t"}, resume=False)
        st = RunStore.create(os.path.join(d, "c"), {"name": "c"})
        for i in range(CHUNK + 10):
            st.add(FrameRef("s", i), {"m": {"x": np.ones(2)}}, {}, {}, [], {})
        del st                                                          # no close(): the last 10 frames never reached the disk
        assert len(RunStore.open(os.path.join(d, "c")).counts("m")) == CHUNK
        fp = os.path.join(d, "c", "frames.jsonl")                       # a crash inside a flush: readers ignore the cut-off last line, a writer cuts it away
        with open(fp, "a", encoding="utf-8") as f:
            f.write('{"label": "s/999", "scene": "s", "ind')
        assert len(RunStore.open(os.path.join(d, "c")).frames) == CHUNK and open(fp, encoding="utf-8").read().endswith("ind"), "reading changes nothing"
        again = RunStore.create(os.path.join(d, "c"), {"name": "c"})
        assert len(again.frames) == CHUNK and not open(fp, encoding="utf-8").read().endswith("ind")
        again.add(FrameRef("s", 999), {"m": {"x": np.ones(2)}}, {}, {}, [], {})
        again.close()
        assert RunStore.open(os.path.join(d, "c")).frames[-1]["label"] == "s/999"
        for tail, why in (("12", "not a frame record"), ("{}\n", "not a frame record"), ("garbage\n", "not valid JSON")):
            with open(fp, "a", encoding="utf-8") as f:
                f.write(tail)
            raises(ValueError, why, RunStore.create, os.path.join(d, "c"), {"name": "c"})
            raw = open(fp, "rb").read()
            assert raw.endswith(tail.encode()), "refused, so not changed"
            open(fp, "wb").write(raw[:-len(tail)])
        lines = open(fp, encoding="utf-8").read().split("\n")
        lines[3] = lines[3][:20]                                         # a damaged line that is not the last one
        open(fp, "w", encoding="utf-8").write("\n".join(lines))
        raises(ValueError, "line 4", RunStore.open, os.path.join(d, "c"))


@check("run identity: a new description key is tolerated with a warning, another value of a key and another metric contract are another run; `ignore` lifts only the named key")
def _():
    class Signed(World):                                                # the adapter after a change: one more key in describe()
        sig = "abc"

        def describe(self):
            return dict(super().describe(), gt_signature=self.sig)

    class Other(Signed):
        sig = "xyz"
    with tempfile.TemporaryDirectory() as d:
        make(d).quant.run(name="a", progress=0)
        new = EvalSuite(Signed(), Smooth(), ToyTask(), out=d)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            st = new.quant.run(name="a", progress=0)                    # resume: accepted
            new.quant.run(name="b", progress=0)
            assert new.quant.compare("a", "b", boot=10)["n_common_frames"] == 32
        assert "gt_signature" in err.getvalue() and "predates" in err.getvalue(), err.getvalue()
        assert "gt_signature" not in st.manifest["source"] and st.manifest["predates_keys"] == ["source: gt_signature"]
        other = EvalSuite(Other(), Smooth(), ToyTask(), out=d)
        raises(ValueError, "gt_signature", other.quant.run, name="b", progress=0)
        other.quant.run(name="c", progress=0)
        raises(ValueError, "gt_signature", other.quant.compare, "b", "c", boot=10)
        old = {"metrics": {"iou": {"headline": ["iou"], "better": {"iou": "high"}}}}
        flipped = {"metrics": {"iou": {"headline": ["iou"], "better": {"iou": "low"}}}}
        more = {"metrics": {**old["metrics"], "ap": {"headline": ["ap"], "better": {"ap": "high"}}}}
        assert len(_identity_differences(old, flipped)) == 1 and _identity_differences(old, {}) == [] == _identity_differences({}, old) == _identity_differences(old, more)
        m1 = {"name": "x", "state": "reset", "source": {"dataset": "s", "evaluator_hash": "aaa"}, "subject": {}, "task": "t", "selection": {}}
        RunStore.create(os.path.join(d, "x"), m1).close()
        m2, ignore = dict(m1, source={"dataset": "s", "evaluator_hash": "bbb"}), {"source": ("evaluator_hash",)}
        raises(ValueError, "evaluator_hash", RunStore.create, os.path.join(d, "x"), m2)
        RunStore.create(os.path.join(d, "x"), m2, ignore=ignore).close()
        raises(ValueError, "dataset", RunStore.create, os.path.join(d, "x"), dict(m2, source={"dataset": "other", "evaluator_hash": "bbb"}), ignore=ignore)


# ------------------------------------------------------------------------------------------------ statistics
@check("stats: robust z (MAD = 0 fallback), leave-one-out by hand, the bootstrap on a planted difference (direction-aware) and on identical runs")
def _():
    v = np.array([1.0, 2, 3, 4, 100, 3, 2, 3])
    med = np.median(v)
    assert np.allclose(robust_z(v), (v - med) / (1.4826 * np.median(np.abs(v - med)))) and robust_z(v).argmax() == 4
    flat = robust_z([5, 5, 5, 5, 9])
    assert flat[4] > 3 and np.allclose(flat[:4], flat[0]) and (robust_z([2, 2, 2]) == 0).all() and np.isnan(robust_z([np.nan, 1.0])[0])
    units = {"a": [{"x": np.array(1.0)}, {"x": np.array(2.0)}], "b": [{"x": np.array(10.0)}], "c": [{"x": np.array(3.0)}]}
    red = lambda cs, keys=None: {"mean": float(np.mean([c["x"] for c in cs])) if cs else float("nan")}          # noqa: E731
    infl = loo_influence(units, red)
    assert abs(infl["b"]["mean"] - (4.0 - 2.0)) < 1e-12 and abs(infl["a"]["mean"] - (4.0 - 6.5)) < 1e-12
    rng = np.random.default_rng(0)
    a = {f"s{i}": [{"x": np.array(rng.normal(0.5, 0.1))} for _ in range(4)] for i in range(12)}
    b = {k: [{"x": c["x"] + 0.2} for c in v] for k, v in a.items()}
    out = bootstrap_scenes(a, b, red, ["mean"], n=300, seed=1)["mean"]          # a = control, b = variant: diff = b - a
    assert abs(out["diff"] - 0.2) < 1e-9 and out["lo"] > 0 and out["p_b_better"] == 1.0 and out["n_valid"] == out["n_draws"] == 300, out
    low = bootstrap_scenes(a, b, red, ["mean"], n=100, seed=1, better={"mean": "low"})["mean"]
    assert low["p_b_better"] == 0.0 and low["p_diff_gt0"] == 1.0, "for a lower-is-better number a higher b is never better"
    same = bootstrap_scenes(a, a, red, ["mean"], n=200, seed=1)["mean"]
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0 and same["p_b_better"] == 0.0, same
    raises(ValueError, "no scene in common", bootstrap_scenes, {"x": []}, {"y": []}, red, ["mean"])


@check("stats: the ONE scene bootstrap - the draw stream is the lane project's (RandomState randint, = choice), paired a - b, non-finite draws counted (n_valid), direction-aware share")
def _():
    assert list(next(scene_picks(5, 1, 0))) == [4, 0, 3, 3, 3], "np.random.RandomState(0).randint(0, 5, 5): the stream every stored lane interval rests on"
    scenes = ["s-07", "s-01", "s-07", "s-30", "s-01", "s-12", "s-30", "s-30"]

    def lane_draws_2026(scenes, B, seed):                              # sparse_lanes/bootstrap.py._scene_draws as it was before D24, verbatim
        scenes = np.asarray(scenes)
        ids, inv = np.unique(scenes, return_inverse=True)
        members = [np.nonzero(inv == s)[0] for s in range(len(ids))]
        rng = np.random.RandomState(seed)
        for _ in range(B):
            pick = rng.randint(0, len(ids), len(ids))
            yield np.concatenate([members[s] for s in pick])
    assert all(np.array_equal(x, y) for x, y in zip(scene_draws(scenes, 50, 3), lane_draws_2026(scenes, 50, 3)))
    us, rng = np.unique(scenes), np.random.RandomState(4)                # topo_metrics.paired drew with choice(): the same stream
    assert all(np.array_equal(us[p], rng.choice(us, len(us), replace=True)) for p in scene_picks(len(us), 50, 4))
    # paired a - b on frame values; key 'far' has no number when scene 0 is not drawn (like R@1 far without far points)
    va, vb = np.arange(1.0, 7.0), np.arange(1.0, 7.0) + 0.5
    sc = [f"s{i}" for i in range(6)]

    def stat(v):
        return lambda idx: {"mean": float(v[idx].mean()) if idx is not None else float(v.mean()),
                            "far": float(v[0]) if idx is None or 0 in idx else float("nan")}
    res = paired_draws(stat(va), stat(vb), scene_draws(sc, 400, 2), ["mean", "far"], better={"mean": "low"})
    m, f = res["mean"], res["far"]
    assert m["diff"] == -0.5 and abs(m["lo"] + 0.5) < 1e-12 and abs(m["hi"] + 0.5) < 1e-12 and m["n_valid"] == 400, m
    assert m["p_gt0"] == 0.0 and m["p_a_better"] == 1.0, "a is lower in every draw: better for a 'low' number"
    with_0 = sum(0 in p for p in scene_picks(6, 400, 2))
    assert f["n_valid"] == with_0 < f["n_draws"] == 400 and f["lo"] == f["hi"] == -0.5 and f["p_a_better"] == 0.0, (f, with_0)
    lo, hi, nv = percentile_interval([np.nan, 1.0, 2.0, np.inf, 3.0])
    assert (lo, hi, nv) == (float(np.percentile([1, 2, 3], 2.5)), float(np.percentile([1, 2, 3], 97.5)), 3)
    assert np.isnan(percentile_interval([np.nan])[0]) and percentile_interval([np.nan])[2] == 0
    iv = interval_draws(stat(va), scene_draws(sc, 300, 1), ["mean", "far"])
    assert iv["mean"]["value"] == 3.5 and iv["mean"]["lo"] < 3.5 < iv["mean"]["hi"] and iv["far"]["n_valid"] < 300 and iv["far"]["sd"] == 0.0


# ------------------------------------------------------------------------------------------------ the frame loop and the counting
@check("runner: carry keeps the memory along a scene and starts each scene empty, reset does not, the full history of one frame equals the carried run; a failing loader raises")
def _():
    src, subj = World(), Smooth()
    carry = [p.taps["memory"] for _, p in Runner(src, subj, "carry").iter_scene("toy-1", taps=["memory"])]
    reset = [p.taps for _, p in Runner(src, subj, "reset").iter_scene("toy-1", taps=["memory", "blur"])]
    assert all(np.array_equal(r["memory"], r["blur"]) for r in reset) and np.array_equal(carry[0], reset[0]["memory"]) and not np.allclose(carry[5], reset[5]["memory"])
    assert np.allclose(carry[4], 0.5 * carry[3] + 0.5 * reset[4]["blur"], atol=1e-6), "the average of the toy network"
    ref = src.frames("toy-1")[6]
    assert np.array_equal(Runner(src, subj, "carry").run_to(ref, ["memory"], history="full")[1].taps["memory"], carry[6])
    assert np.array_equal(Runner(src, subj, "carry").run_to(ref, ["memory"], cold=True)[1].taps["memory"], reset[6]["memory"])

    class Bad(World):
        def load(self, ref):
            if ref.index == 3:
                raise RuntimeError("disk gone")
            return super().load(ref)
    raises(RuntimeError, "disk gone", lambda: list(Runner(Bad(), Smooth(), "carry").iter_scene("toy-0")))
    for i, _ in enumerate(Runner(World(), Smooth(), "carry").iter_scene("toy-0")):      # breaking out of the loop does not hang
        if i == 2:
            break


@check("quant.run: the numbers are those computed without the engine; carry refuses a gap; another frame list or other frame keys under one name are refused; an interrupted run resumes")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        tab = suite.quant.table(st)
        assert len(st.frames) == 32 and st.manifest["finished"] and abs(tab["seg.iou"] - direct_iou(World(), Smooth())) < 1e-12
        assert abs(suite.quant.per_scene(st)["toy-3"]["seg.iou"] - direct_iou(World(), Smooth(), ["toy-3"])) < 1e-12
        reset = suite.quant.table(suite.quant.run(name="r", state="reset", progress=0))
        assert abs(reset["seg.iou"] - direct_iou(World(), Smooth(), reset=True)) < 1e-12 and reset["seg.iou"] != tab["seg.iou"]
        gap = Sampler(suite.source, 0).spread(10)
        raises(ValueError, "consecutive frames", suite.quant.run, frames=gap, name="g", progress=0)
        assert {f.label for f in gap} == set(suite.quant.run(frames=gap, state="reset", name="g", progress=0).labels())
        raises(ValueError, "no frames", suite.quant.run, frames=[], name="empty", progress=0)
        assert not os.path.exists(os.path.join(d, "quant", "empty")), "nothing is stored for a refused run"
        for name, first, other, key in (("pick", FrameRef("toy-1", 0), FrameRef("toy-1", 3), "frames_sha"), ("keyed", FrameRef("toy-1", 0, "tokA"), FrameRef("toy-1", 0, "tokB"), "keys_sha")):
            for _ in range(2):                                          # the same selection resumes
                assert len(suite.quant.run(frames=[first], state="reset", name=name, progress=0).frames) == 1
            raises(ValueError, key, suite.quant.run, frames=[other], state="reset", name=name, progress=0)

        class Flaky(World):
            boom = True

            def load(self, ref):
                if Flaky.boom and ref.scene == "toy-2" and ref.index == 4:
                    raise RuntimeError("crash")
                return super().load(ref)
        flaky = EvalSuite(Flaky(), Smooth(), ToyTask(), out=d)
        raises(RuntimeError, "crash", flaky.quant.run, name="resumed", progress=0)
        part = RunStore.open(os.path.join(d, "quant", "resumed"))
        assert not part.manifest["finished"] and 0 < len(part.frames) < 32
        Flaky.boom = False
        done = flaky.quant.run(name="resumed", progress=0)
        assert len(done.frames) == 32 and len(set(done.labels())) == 32 and done.manifest["finished"] and flaky.quant.table(done) == tab, "the numbers of an undisturbed run"


@check("quant.run: n_scenes and max_frames choose subsets; state 'warmup' starts every frame empty and feeds only the network's warm-up history")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        some = suite.quant.run(scenes=["toy-1", "toy-2"], max_frames=5, name="some", progress=0)
        assert some.labels() == [f"toy-{s}/{i:03d}" for s in (1, 2) for i in range(5)] and some.manifest["selection"]["max_frames"] == 5
        drawn = suite.quant.run(n_scenes=3, seed=1, name="drawn", progress=0)
        assert drawn.scene_names() == Sampler(suite.source, 1).scenes(3) and len(drawn.frames) == 24
        frames = Sampler(suite.source, 0).spread(8, min_index=3)
        assert len(frames) == 8 and all(f.index >= 3 for f in frames)
        warm = suite.quant.run(frames=frames, state="warmup", name="warm", progress=0)          # any subset of frames is allowed
        src, net, metric = World(), Smooth(), ToyTask().metrics()[0]
        for ref in frames:                                              # by hand: empty, the two frames before (Smooth.warmup), the frame
            net.reset()
            history = src.frames(ref.scene)[ref.index - 2:ref.index]
            want = [net.predict(src.load(r)) for r in history + [ref]][-1]
            assert np.array_equal(metric.frame(src.load(ref), want)["c"], warm.counts("seg", [ref.label])[0]["c"]), ref.label
        cold = suite.quant.run(frames=frames, state="reset", name="cold", progress=0)
        assert suite.quant.table(cold) != suite.quant.table(warm), "the warm-up history changes the numbers of a network with a memory"
        for gone, word in ((suite.quant.trace, "draw_frame"), (suite.quant.pareto, "get_cost")):
            raises(NotImplementedError, word, gone, "warm")


@check("outliers and report: the planted frames come first, a scene's drag is leave-one-out by hand, support keeps empty frames out; rows name the draw-frame command; the page links resolve")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        osr = suite.quant.outliers(st, k=5)
        planted = {f"{s}/{i:03d}" for s, i in PLANTED}
        for metric in ("seg.iou", "wrong.cells"):                       # one higher is better, one lower
            rows = [r for r in osr["frames"] if r["metric"] == metric]
            assert planted <= {r["label"] for r in rows}, (metric, [r["label"] for r in rows])
            assert all(r["outlier"] and all(v > 0 for v in r["drag"].values()) for r in rows if r["label"] in planted), rows
        scenes = [r for r in osr["scenes"] if r["metric"] == "seg.iou"]
        m, counts = suite.task.metrics()[0], st.counts("seg")
        without = m.reduce([c for fr, c in zip(st.frames, counts) if fr["scene"] != "toy-3"])["iou"]
        assert scenes[0]["scene"] == "toy-3" and abs(scenes[0]["drag"] - (without - m.reduce(counts)["iou"])) < 1e-12 and scenes[0]["drag"] > 0
        assert [r["drag"] for r in scenes] == sorted((r["drag"] for r in scenes), reverse=True)
        assert scenes[0]["view"] == "draw-frame ID toy-3/000" and osr["frames"][0]["view"] == f"draw-frame ID {osr['frames'][0]['label']}"
        assert suite.quant.outliers(st, k=50, min_support=10 ** 9)["frames"] == []
        path = suite.quant.report(st)
        text = open(path, encoding="utf-8").read()
        assert "toy-3" in text and "headline numbers" in text and "draw-frame" in text and "qual frame" not in text and report.links_in(path) == []
        rows = open(os.path.join(st.path, "frames.csv")).read().strip().split("\n")
        assert len(rows) == 33 and "seg.iou" in rows[0]
        page = report.write(os.path.join(d, "sub", "i.html"), "t <&>", [("h", report.table([["k", "v"], ["w", "warn"]], status_col=1)),
                                                                         ("", report.image(path, os.path.join(d, "sub"), 100))])
        assert "t &lt;&amp;&gt;" in open(page, encoding="utf-8").read() and 'class="warn"' in open(page, encoding="utf-8").read() and len(report.links_in(page)) == 2


@check("a metric that returns {} for a whole chunk (nothing to count there) still gives tables, compare and outliers")
def _():
    class Sparse(Wrong):
        name = "sparse"

        def frame(self, sample, pred):
            return super().frame(sample, pred) if sample.ref.scene == "s1" else {}

        def reduce(self, counts, keys=None):
            return super().reduce([c for c in counts if c])

        def score(self, counts):
            return super().score(counts) if counts else {}

    class Long(World):
        def scenes(self):
            return ["s0", "s1", "s2"]

        def frames(self, scene):
            return [FrameRef(scene, t) for t in range(CHUNK)]

        def load(self, ref):
            return Sample(ref, inputs=np.eye(8, dtype=np.float32) * (ref.index % 3), reference=np.zeros((8, 8), bool))

    class Both(ToyTask):
        def metrics(self):
            return [Wrong(), Sparse()]
    with tempfile.TemporaryDirectory() as d:
        suite = EvalSuite(Long(), Smooth(-0.4), Both(), out=d)
        st = suite.quant.run(name="a", progress=0)                      # chunk 0 = s0 (all {}), chunk 1 = s1, chunk 2 = s2 ({})
        assert all(os.path.exists(os.path.join(st.path, f"counts_sparse_{k:05d}.npz")) for k in range(3))
        os.remove(os.path.join(st.path, "counts_sparse_00000.npz"))     # a run stored before empty chunks got a file
        want = Wrong().reduce(suite.quant.open("a").counts("wrong", [f.label for f in suite.source.frames("s1")]))["cells"]
        assert want > 0 and suite.quant.table("a")["sparse.cells"] == want
        assert suite.quant.compare("a", "a", boot=20)["metrics"]["sparse.cells"]["diff"] == 0 and suite.quant.outliers("a", k=2)["scenes"]


@check("compare: a worse network is detected (b - a < 0, the interval below 0), a run against itself gives 0; another source or another frame key is refused unless forced")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        suite.quant.run(name="a", progress=0)
        make(d, bias=0.3).quant.run(name="b", progress=0)
        cmp = suite.quant.compare("a", "b", boot=300)
        iou = cmp["metrics"]["seg.iou"]
        assert iou["diff"] < -0.02 and iou["hi"] < 0 and iou["p_b_better"] == 0.0 and iou["n_valid"] == 300 and cmp["n_common_frames"] == 32, iou
        same = suite.quant.compare("a", "a", boot=100)["metrics"]["seg.iou"]
        assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
        make(d, seed=1).quant.run(name="o", progress=0)
        raises(ValueError, "same source and task", suite.quant.compare, "a", "o", boot=20)
        assert suite.quant.compare("a", "o", boot=20, force=True)["n_common_frames"] == 32
        make(d).quant.run(scenes=["toy-0", "toy-1"], name="p", progress=0)
        assert suite.quant.compare("p", "a", boot=50)["n_common_frames"] == 16, "the frames both runs have"
        for name, key in (("ka", "tokA"), ("kb", "tokB")):              # the same label over another frame key is another frame
            make(d).quant.run(frames=[FrameRef("toy-0", 0, key)], name=name, progress=0)
        raises(ValueError, "frame key", suite.quant.compare, "ka", "kb", boot=20)
        assert suite.quant.compare("ka", "kb", boot=20, force=True)["n_common_frames"] == 1


# ------------------------------------------------------------------------------------------------ drawing
@check("figure: the error colours at the right cells (TP grey, FP grey-green, FN grey-blue, not evaluated cream) and their counts")
def _():
    p = np.array([[1, 1], [0, 0]], bool)
    g = np.array([[1, 0], [1, 0]], bool)
    e = figure.error_rgb(p, g)
    ac = np.allclose
    assert ac(e[0, 0], figure.TP_COLOR) and ac(e[0, 1], figure.FP_COLOR) and ac(e[1, 0], figure.FN_COLOR) and ac(e[1, 1], figure.TN_COLOR)
    v = figure.error_rgb(p, g, valid=np.array([[1, 1], [1, 0]], bool))
    assert ac(v[1, 1], figure.SKIP_COLOR) and figure.error_counts(p, g) == (1, 1, 1)
    img = figure.render_page(Page("t", [Panel("rgb", "e", e, extent=(0, 2, 0, 2))], ncols=1, panel_size=(3, 3)))
    near = lambda c: (np.abs(img.astype(float) - np.array(c) * 255).max(-1) < 3)                          # noqa: E731
    pos = {n: np.argwhere(near(c)).mean(0) for n, c in (("tp", figure.TP_COLOR), ("fp", figure.FP_COLOR), ("fn", figure.FN_COLOR))}
    assert all(near(c).sum() > 200 for c in (figure.TP_COLOR, figure.FP_COLOR, figure.FN_COLOR)), "all three colours are drawn"
    # array[0, 0] is the lower left cell (origin lower): TP left of FP on the same row, FN above TP
    assert pos["fp"][1] > pos["tp"][1] + 20 and abs(pos["fp"][0] - pos["tp"][0]) < 5 and pos["fn"][0] < pos["tp"][0] - 20, pos


@check("figure: every panel kind and overlay draws, also boolean maps and notes; the page follows the grid; PNG and array agree")
def _():
    from PIL import Image
    a = np.random.default_rng(0).random((20, 30))
    shapes = [Shape("polygon", [(2, 2), (8, 2), (8, 8)], "r"), Shape("line", [(0, 0), (10, 10)]), Shape("points", [(5, 5), (6, 7)]),
              Shape("circle", (10, 10, 3)), Shape("arrow", (1, 1, 6, 6)), Shape("text", (5, 5), label="x")]
    panels = [Panel("image", "i", (a * 255).astype(np.uint8)), Panel("raster", "r", a, extent=(0, 30, 0, 20), colorbar=True, shapes=shapes, xlim=(30, 0), note="what to look for"),
              Panel("rgb", "c", np.dstack([a, a, a]), extent=(0, 30, 0, 20)), Panel("text", "t", "hello\nworld"), Panel("table", "tab", [["a", "b"], [1, 2.5]]),
              Panel("bars", "b", {"x": 1.0, "y": 2.0})]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.png")
        img = figure.render_page(Page("title\nsecond line", panels, ncols=3, legend=figure.ERROR_LEGEND, footer="foot"), path, dpi=60)
        assert Image.open(path).size == (img.shape[1], img.shape[0]) and abs(img.shape[1] / 60 - 3 * 4.6) < 0.1 and img.shape[0] / 60 > 2 * 4.2
    draw = lambda panel: figure.render_page(Page("t", [panel]))                                           # noqa: E731
    assert draw(Panel("raster", "mask", a > 0.5)).ndim == 3, "a boolean map is drawn"
    for kind, data in (("image", np.zeros((16, 16, 3), np.uint8)), ("bars", {"a": 1.0}), ("text", "hello")):
        assert not np.array_equal(draw(Panel(kind, "p", data)), draw(Panel(kind, "p", data, note="what to look for"))), f"the note of a {kind} panel is drawn"
    raises(ValueError, "unknown panel kind", draw, Panel("nope"))


@check("taps: energy, PCA colours, panels per kind, the statistics table, plausibility flags (dead, constant, NaN, no structure, magnitude)")
def _():
    a = np.ones((3, 4, 5), np.float32)
    assert np.allclose(tapmod.energy(a), np.sqrt(3))
    c = tapmod.pca_rgb(np.random.default_rng(1).random((5, 8, 8)))
    assert c.shape == (8, 8, 3) and c.min() >= 0 and c.max() <= 1
    broken = np.random.RandomState(0).randn(4, 6, 6).astype(np.float32)
    broken[0, 0, 0], broken[1, 1, 1] = np.nan, np.inf
    assert np.isfinite(tapmod.pca_rgb(broken)).all() and np.isfinite(tapmod.energy(broken)).all(), "a broken tensor is flagged, its picture is still drawn"
    rng = np.random.default_rng(0)
    specs = [TapSpec("fm", "feature_map", "s1"), TapSpec("lg", "logits"), TapSpec("im", "image"), TapSpec("mp", "map"), TapSpec("vec", "vector", note="n", redflag="all equal"),
             TapSpec("sc", "scalar")]
    taps = {"fm": rng.random((4, 6, 6)), "lg": rng.normal(size=(2, 6, 6)), "im": rng.random((6, 6, 3)), "mp": rng.random((6, 6)), "vec": np.arange(40.0), "sc": np.array(3.0)}
    panels = tapmod.tap_panels(taps, specs)
    assert [p.kind for p in panels] == ["raster", "rgb", "raster", "image", "raster", "bars", "text"] and panels[5].note == "n; red flag: all equal; first 16 of 40 values"
    assert [p.title for p in tapmod.tap_panels(taps, specs, wanted=["mp"])] == ["mp"]
    flags = lambda arr: {f.name.split(": ")[1]: f.status for f in tapmod.plausibility(TapSpec("t"), arr)}       # noqa: E731
    assert flags(np.zeros((2, 4, 4))) == {"alive": "fail"} and flags(np.full((2, 4, 4), 3.0)).get("alive") == "warn" and flags(broken)["finite"] == "fail"
    assert flags(rng.random((2, 4, 4))) == {} and flags(np.full((2, 4, 4), 1e9) + rng.random((2, 4, 4)))["magnitude"] == "warn"
    assert tapmod.plausibility(TapSpec("t", alive_check=False), np.zeros((2, 4, 4))) == []
    table = tapmod.stats_table({"a": a}, [Flag("a: alive", "fail", 0, "dead")])
    assert table.kind == "table" and any("FAIL" in str(r) for r in table.data)


# ------------------------------------------------------------------------------------------------ torch helpers (skipped without torch)
@check("torch_taps: hooks by module path (out / in / first / list / pick), arming, an optional module that is not there, a tap that cannot be read warns")
def _():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("      (torch not installed: skipped)")
        return
    from ..torch_taps import Hook, TapRecorder

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(3, 4), nn.Linear(4, 2)
            self.opt = None

        def forward(self, x):
            return self.b(self.a(x)) + self.b(self.a(x * 2))                     # a and b are called twice
    m = M()
    rec = TapRecorder(m, {"a_out": Hook("a"), "a_in": Hook("a", when="in"), "a_first": Hook("a", select="first"),
                          "a_list": Hook("a", select="list"), "b_pick": Hook("b", pick=lambda o: o[:, :1]), "gone": Hook("opt")})
    x = torch.randn(2, 3)
    m(x)
    assert rec.read() == {}, "nothing is recorded unless armed"
    rec.arm(["a_out", "a_in", "a_first", "a_list", "b_pick"])
    with torch.no_grad():
        m(x)
    got = rec.read(0)
    assert np.allclose(got["a_in"], x[0].numpy() * 2), "'in' with select last = the second call's input"
    assert np.allclose(got["a_first"], m.a(x)[0].detach().numpy(), atol=1e-6) and got["a_out"].shape == (4,)
    assert got["a_list"].shape == (2, 4) and got["b_pick"].shape == (1,) and rec.read(None)["a_out"].shape == (2, 4)
    raises(KeyError, "unknown taps", rec.arm, ["nope"])
    rec.close()

    class K(nn.Module):                                                       # keyword-only call, an in-place layer after the tapped one
        def __init__(self):
            super().__init__()
            self.c, self.r = nn.Conv1d(1, 1, 1), nn.ReLU(inplace=True)

        def forward(self, feat):
            return self.r(self.c(input=feat))                                 # the tapped module gets its input by keyword
    k = K()
    with torch.no_grad():
        k.c.weight.fill_(-1.0)
        k.c.bias.zero_()
    rec = TapRecorder(k, {"in": Hook("c", when="in"), "conv": Hook("c")})
    rec.arm(["in", "conv"])
    with torch.no_grad():
        k(torch.ones(1, 1, 5))                                                # no positional argument for the pre-hook: the tap is skipped, the forward pass runs
    with warnings.catch_warnings(record=True) as caught:                      # a failing tap is not a silent gap: read() names it and its count
        warnings.simplefilter("always")
        got = rec.read()
    assert rec.errors.get("in") == 1 and rec.failed == {"in": 1} and "in" not in got, (rec.errors, rec.failed, got.keys())
    assert len(caught) == 1 and issubclass(caught[0].category, RuntimeWarning) and "'in': 1" in str(caught[0].message), [str(w.message) for w in caught]
    assert got["conv"].min() == -1.0, "the record is the output of the tapped layer, not what an in-place ReLU made of it"
    rec.arm(["conv"])                                                         # a new pass starts clean: no failure, no warning (errors keeps the total)
    with torch.no_grad():
        k(torch.ones(1, 1, 5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec.read()
    assert rec.failed == {} and rec.errors == {"in": 1} and not caught, (rec.failed, rec.errors, [str(w.message) for w in caught])
    rec.close()


@check("torch_taps.count_ops: a small network against the hand count (convolutions and linear layers; ReLU and flatten add nothing), parts down to a depth")
def _():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("      (torch not installed: skipped)")
        return
    from ..torch_taps import count_ops
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1), nn.Flatten(), nn.Linear(4 * 16 * 16, 10))
    x = torch.zeros(1, 3, 16, 16)
    r = count_ops(lambda: m(x))
    conv1, conv2, lin = 16 * 16 * 8 * 3 * 9, 16 * 16 * 4 * 8, 1024 * 10
    assert round(r["gmacs"] * 1e9) == conv1 + conv2 + lin and r["method"] == "measured" and "not counted" in r["note"], r
    assert [round(r["parts"][f"Sequential.{i}"] * 1e9) for i in (0, 2, 4)] == [conv1, conv2, lin] and "Global" not in r["parts"] and "Sequential" not in r["parts"]
    assert round(count_ops(lambda: (m(x), m(x)))["gmacs"] * 1e9) == 2 * (conv1 + conv2 + lin), "two calls are counted twice"

    class Deep(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Sequential(nn.Linear(8, 8)))
            self.b = nn.Linear(8, 8)

        def forward(self, z):
            return self.b(self.a(z))
    parts = lambda depth: count_ops(lambda: Deep()(torch.zeros(1, 8)), depth=depth)["parts"]            # noqa: E731
    assert set(parts(1)) == {"Deep.a", "Deep.b"} and set(parts(3)) == {"Deep.a", "Deep.a.0", "Deep.a.0.0", "Deep.b"}
    assert all(round(v * 1e9) == 64 for v in parts(3).values()), "a nested module is also counted in its parent"

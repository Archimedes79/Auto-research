"""Known-answer tests of every function of the suite, on the toy adapter (numpy, matplotlib; torch only for the torch_taps test).

    python -m evalsuite selftest            (from autoresearch/)      or      python autoresearch/evalsuite/selftest.py

Every check states what it proves; a failing check prints its traceback. Each function of contracts / sampling / store / stats /
runner / qualitative / quantitative / probing / render / check / cli is covered; the planted outlier frames of the toy world are the ground
truth of the outlier search.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

if __package__ in (None, ""):                                          # run as a script
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    __package__ = "evalsuite"

from evalsuite import EvalSuite, FrameRef, Page, Panel, Prediction, Sample, TapSpec                 # noqa: E402
from evalsuite.adapters import toy                                                                  # noqa: E402
from evalsuite.check import check_adapter                                                           # noqa: E402
from evalsuite.contracts import Flag, Source                                                        # noqa: E402
from evalsuite.render import figure, report, taps as tapmod, video                                  # noqa: E402
from evalsuite.sampling import Sampler                                                              # noqa: E402
from evalsuite.stats import bootstrap_scenes, iqr_fences, loo_influence, robust_z                   # noqa: E402
from evalsuite.store import CHUNK, RunStore                                                         # noqa: E402

CHECKS: List[Tuple[str, Callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make(tmp: str, **kw) -> EvalSuite:
    state = kw.pop("state", "carry")
    src, subj, task = toy.build(**kw)
    return EvalSuite(src, subj, task, out=tmp, state=state)


def direct_seg(src, subj, scenes=None, reset=False):
    """the toy IoU computed without the suite: every frame through the subject, cells summed"""
    tp = fp = fn = 0
    for sc in (scenes or src.scenes()):
        subj.reset()
        for ref in src.frames(sc):
            if reset:
                subj.reset()
            s = src.load(ref)
            p = subj.predict(s).outputs["mask"]
            g = s.reference["mask"]
            tp, fp, fn = tp + (p & g).sum(), fp + (p & ~g).sum(), fn + (~p & g).sum()
    return tp / (tp + fp + fn)


# ------------------------------------------------------------------------------------------------ contracts and sampling
@check("FrameRef: label, slug, parse")
def _():
    r = FrameRef("scene-0061", 12, "tok")
    assert r.label == "scene-0061/012" and r.slug == "scene-0061_012" and str(r) == r.label
    assert FrameRef.parse("scene-0061/12") == ("scene-0061", 12) and FrameRef.parse("scene-0061:012") == ("scene-0061", 12)
    assert FrameRef.parse("scene-0061") is None and FrameRef.parse("a/b") is None
    assert FrameRef("a", 1, "x") == FrameRef("a", 1, "y"), "the key is not part of the identity"


@check("Sample.get: lazy display values are computed once")
def _():
    calls = []
    s = Sample(FrameRef("a", 0), None, None, {"img": lambda: calls.append(1) or np.ones(2)})
    assert s.get("img").sum() == 2 and s.get("img").sum() == 2 and len(calls) == 1 and s.get("none", 5) == 5


@check("Source.find: label, tuple, FrameRef, key; errors name the closest scenes")
def _():
    src = toy.ToySource()
    f = src.frames("toy-02")[7]
    assert src.find("toy-02/7") == f and src.find(("toy-02", 7)) == f and src.find(f) == f and src.find(f.key) == f
    for bad in ("toy-99/1", "toy-02/99", "nonsense"):
        try:
            src.find(bad)
            raise AssertionError(bad)
        except KeyError:
            pass
    try:
        src.find("toy-9/1")
    except KeyError as e:
        assert "toy-0" in str(e) or "closest" in str(e)


class _Uneven(Source):
    """two scenes: A has 2 frames, B has 10 (to see the video-first law)"""
    def scenes(self): return ["A", "B"]
    def frames(self, s): return [FrameRef(s, i) for i in range(2 if s == "A" else 10)]
    def load(self, ref): return Sample(ref, None)


@check("Sampler: video first, then frame - every video equally likely, every frame of a video equally likely")
def _():
    sm = Sampler(_Uneven(), seed=5)
    draws = sm.frames(6000, unique=False)
    pa = np.mean([d.scene == "A" for d in draws])
    assert abs(pa - 0.5) < 4 * np.sqrt(0.25 / 6000), pa                       # 4 sigma around 1/2
    ia = np.bincount([d.index for d in draws if d.scene == "A"], minlength=2) / sum(d.scene == "A" for d in draws)
    assert np.abs(ia - 0.5).max() < 0.05, ia
    ib = np.bincount([d.index for d in draws if d.scene == "B"], minlength=10) / sum(d.scene == "B" for d in draws)
    assert np.abs(ib - 0.1).max() < 0.03, ib


@check("Sampler: seeded, no repeats, errors, scenes(m), spread(n)")
def _():
    src = toy.ToySource()
    a, b = Sampler(src, 3).frames(7), Sampler(src, 3).frames(7)
    assert [x.label for x in a] == [x.label for x in b] and len({x.label for x in a}) == 7
    assert [x.label for x in Sampler(src, 4).frames(7)] != [x.label for x in a]
    try:
        Sampler(src, 0).frames(10 ** 6)
        raise AssertionError("must raise")
    except ValueError:
        pass
    sc = Sampler(src, 1).scenes(3)
    assert len(sc) == 3 and len(set(sc)) == 3 and len(Sampler(src, 1).scenes(99)) == 8
    sp = Sampler(src, 0).spread(10)
    flat = [f for s in src.scenes() for f in src.frames(s)]
    assert [f.label for f in sp] == [flat[i].label for i in sorted({int(i) for i in np.linspace(0, 95, 10).round()})]
    assert len(Sampler(src, 0).spread(1000)) == 96


# ------------------------------------------------------------------------------------------------ store and stats
@check("RunStore: round trip over several chunks (fixed and ragged counts), resume, crash leaves a consistent prefix")
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
        got = r.counts("m")
        for w, g in zip(want, got):
            assert np.array_equal(w["fixed"], g["fixed"]) and np.array_equal(w["ragged"], g["ragged"]) and g["ragged"].shape == w["ragged"].shape
        assert set(r.by_scene("m")) == {"s0", "s1", "s2"} and sum(len(v) for v in r.by_scene("m").values()) == n
        assert r.frame("s0/003")["key"] == "k3" and len(r.counts("m", ["s0/003", "s1/002"])) == 2
        # resume: the store knows what is done
        r2 = RunStore.create(os.path.join(d, "r"), {"name": "t"}, resume=True)
        assert r2.done() == r.done() and r2.manifest["resumed"] == 1
        try:
            RunStore.create(os.path.join(d, "r"), {"name": "t"}, resume=False)
            raise AssertionError("must raise")
        except FileExistsError:
            pass
        # crash: frames that were never flushed are not in the store
        st = RunStore.create(os.path.join(d, "c"), {"name": "c"})
        for i in range(CHUNK + 10):
            st.add(FrameRef("s", i), {"m": {"x": np.ones(2)}}, {}, {}, [], {})
        del st                                                          # no close(): the last 10 frames never reached disk
        r3 = RunStore.open(os.path.join(d, "c"))
        assert len(r3.frames) == CHUNK and len(r3.counts("m")) == CHUNK
        # a crash inside a flush leaves a cut-off last line: readers ignore it, the store opened for writing cuts it away and goes on
        fp = os.path.join(d, "c", "frames.jsonl")
        with open(fp, "a", encoding="utf-8") as f:
            f.write('{"label": "s/999", "scene": "s", "ind')
        assert len(RunStore.open(os.path.join(d, "c")).frames) == CHUNK, "a cut-off last line is ignored"
        assert open(fp, encoding="utf-8").read().endswith("ind"), "reading does not change the file"
        r4 = RunStore.create(os.path.join(d, "c"), {"name": "c"}, resume=True)
        assert len(r4.frames) == CHUNK and not open(fp, encoding="utf-8").read().endswith("ind")
        r4.add(FrameRef("s", 999), {"m": {"x": np.ones(2)}}, {}, {}, [], {})
        r4.close()
        r5 = RunStore.open(os.path.join(d, "c"))
        assert len(r5.frames) == CHUNK + 1 and r5.frames[-1]["label"] == "s/999" and len(r5.counts("m")) == CHUNK + 1
        d2 = os.path.join(d, "d2")                                         # damage that is not a cut-off record is refused, never repaired away
        st2 = RunStore.create(d2, {"name": "d2"})
        for i in range(3):
            st2.add(FrameRef("s", i), {"m": {"x": np.ones(2)}}, {}, {}, [], {})
        st2.close()
        for tail, why in (("12", "not a frame record"), ("{}\n", "not a frame record"), ("garbage\n", "not valid JSON")):
            with open(os.path.join(d2, "frames.jsonl"), "a", encoding="utf-8") as f:
                f.write(tail)
            try:
                RunStore.create(d2, {"name": "d2"}, resume=True)
                raise AssertionError(f"{tail!r} must be refused")
            except ValueError as e:
                assert why in str(e), e
            raw = open(os.path.join(d2, "frames.jsonl"), "rb").read()
            assert raw.endswith(tail.encode()), "refused, so not changed"
            open(os.path.join(d2, "frames.jsonl"), "wb").write(raw[:-len(tail)])
        lines = open(fp, encoding="utf-8").read().split("\n")
        lines[3] = lines[3][:20]                                         # a damaged line that is not the last one: an error, not a repair
        open(fp, "w", encoding="utf-8").write("\n".join(lines))
        try:
            RunStore.open(os.path.join(d, "c"))
            raise AssertionError("a damaged interior line must be refused")
        except ValueError as e:
            assert "line 4" in str(e), e
        assert r.frame("s0/3") is r.frame("s0/003") is r.frame("s0:3"), "an unpadded label names the same frame"
        try:
            r.frame("s9/0")
            raise AssertionError("an unknown frame must raise")
        except KeyError:
            pass


@check("RunStore / timing / compare: a description key that is NEW (absent from the stored run, e.g. a GT signature added to the adapter) is tolerated with a warning; a key on both sides with another value still refuses")
def _():
    import io
    import contextlib

    class NewSource(toy.ToySource):                                     # the adapter after the change: one more key in describe()
        def __init__(self, sig="abc", **kw):
            super().__init__(**kw)
            self.sig = sig

        def describe(self):
            return dict(super().describe(), gt_signature=self.sig)
    with tempfile.TemporaryDirectory() as d:
        old = EvalSuite(toy.ToySource(n_scenes=2, n_frames=3), toy.ToySubject(), toy.ToyTask(), out=d)
        old.quant.run(name="a", progress=0)                             # written by the old adapter: no gt_signature
        new = EvalSuite(NewSource(n_scenes=2, n_frames=3), toy.ToySubject(), toy.ToyTask(), out=d)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            st = new.quant.run(name="a", progress=0)                    # resume: accepted
            new.quant.timing(n=2, run="a")                              # a benchmark attached to the old store: accepted
            new.quant.run(name="b", progress=0)
            cmp = new.quant.compare("a", "b", boot=10)                  # old store vs new store: accepted
        assert "gt_signature" in err.getvalue() and "predates" in err.getvalue(), err.getvalue()
        assert "gt_signature" not in st.manifest["source"] and st.manifest["predates_keys"] == ["source: gt_signature"]
        assert cmp["n_common_frames"] == 6 and RunStore.open(os.path.join(d, "quant", "a")).manifest["cost"]["bench"]["n"] == 2
        other = EvalSuite(NewSource("xyz", n_scenes=2, n_frames=3), toy.ToySubject(), toy.ToyTask(), out=d)
        for fn in (lambda: other.quant.run(name="b", progress=0), lambda: other.quant.timing(n=2, run="b")):
            try:                                                        # the key on both sides, another value: refused
                fn()
                raise AssertionError("another gt_signature must be refused")
            except ValueError as e:
                assert "gt_signature" in str(e), e
        other.quant.run(name="c", progress=0)
        try:
            other.quant.compare("b", "c", boot=10)
            raise AssertionError("compare of two signatures must be refused")
        except ValueError as e:
            assert "gt_signature" in str(e), e


@check("RunStore + quant: a metric that returns {} for a whole chunk (nothing to count there) still gives tables, compare and outliers (S5)")
def _():
    class Sparse(toy.SegMetric):                                        # counts only in scene toy-01; {} elsewhere
        name, headline = "sparse", ("iou",)

        def frame(self, sample, pred):
            return super().frame(sample, pred) if sample.ref.scene == "toy-01" else {}

        def reduce(self, counts, keys=None):
            return super().reduce([c for c in counts if c])

        def score(self, counts):
            return super().score(counts) if counts else {}

        def support(self, counts):
            return super().support(counts) if counts else {}

    class T(toy.ToyTask):
        def metrics(self):
            return [toy.SegMetric(), Sparse()]
    with tempfile.TemporaryDirectory() as d:
        suite = EvalSuite(toy.ToySource(n_scenes=3, n_frames=CHUNK), toy.ToySubject(), T(), out=d)
        st = suite.quant.run(name="a", progress=0)                     # chunk 0 = toy-00 (all {}), chunk 1 = toy-01, chunk 2 = toy-02 ({})
        assert all(os.path.exists(os.path.join(st.path, f"counts_sparse_{k:05d}.npz")) for k in range(3))
        os.remove(os.path.join(st.path, "counts_sparse_00000.npz"))     # a run stored before the fix: the file is missing
        tab = suite.quant.table("a")
        direct = toy.SegMetric().reduce(suite.quant.open("a").counts("seg", [f.label for f in suite.source.frames("toy-01")]))
        assert abs(tab["sparse.iou"] - direct["iou"]) < 1e-12, (tab, direct)
        assert suite.quant.compare("a", "a", boot=20)["metrics"]["sparse.iou"]["diff"] == 0
        assert suite.quant.outliers("a", k=2)["scenes"]


@check("stats: robust_z, MAD = 0 fallback, IQR fences against numpy")
def _():
    v = np.array([1.0, 2, 3, 4, 100, 3, 2, 3])
    z = robust_z(v)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    assert np.allclose(z, (v - med) / mad) and z.argmax() == 4
    flat = robust_z([5, 5, 5, 5, 9])
    assert flat[4] > 3 and np.allclose(flat[:4], flat[0]), flat
    assert (robust_z([2, 2, 2]) == 0).all() and np.isnan(robust_z([np.nan, 1.0])[0])
    lo, hi = iqr_fences(np.arange(1, 101))
    q1, q3 = np.percentile(np.arange(1, 101), [25, 75])
    assert abs(lo - (q1 - 1.5 * (q3 - q1))) < 1e-9 and abs(hi - (q3 + 1.5 * (q3 - q1))) < 1e-9


@check("stats: leave-one-out influence by hand; bootstrap on a planted difference and on identical runs")
def _():
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
    try:
        bootstrap_scenes({"x": []}, {"y": []}, red, ["mean"])
        raise AssertionError("must raise")
    except ValueError:
        pass


@check("stats: the ONE scene bootstrap - the draw stream is the lane project's (RandomState randint, = choice), paired a - b, non-finite draws counted (n_valid), direction-aware share")
def _():
    from evalsuite.stats import paired_draws, interval_draws, percentile_interval, scene_draws, scene_picks
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


# ------------------------------------------------------------------------------------------------ runner, metrics, state
@check("Runner: carry keeps the memory along a scene, reset does not, run_to(cold) = reset, warm history is closer to carry")
def _():
    from evalsuite.runner import Runner
    src, subj = toy.ToySource(), toy.ToySubject()
    carry = [p.taps["memory"] for _, p in Runner(src, subj, "carry").iter_scene("toy-01", taps=["memory"])]
    reset = [p.taps["memory"] for _, p in Runner(src, subj, "reset").iter_scene("toy-01", taps=["memory"])]
    blur = [p.taps["blur"] for _, p in Runner(src, subj, "reset").iter_scene("toy-01", taps=["blur"])]
    assert np.array_equal(carry[0], reset[0]) and not np.allclose(carry[5], reset[5])
    assert all(np.array_equal(r, b) for r, b in zip(reset, blur)), "reset: memory = blur (nothing carried)"
    assert np.allclose(carry[4], 0.5 * carry[3] + 0.5 * blur[4], atol=1e-6), "the EMA of the toy network"
    run = Runner(src, subj, "carry")
    ref = src.frames("toy-01")[8]
    _, cold = run.run_to(ref, ["memory"], cold=True)
    _, warm = run.run_to(ref, ["memory"])
    assert np.array_equal(cold.taps["memory"], reset[8])
    assert np.abs(warm.taps["memory"] - carry[8]).max() < np.abs(cold.taps["memory"] - carry[8]).max()
    # a new scene starts from an empty memory even in carry mode
    first = [p.taps["memory"] for _, p in Runner(src, subj, "carry").iter_scene("toy-02", taps=["memory"])][0]
    assert np.array_equal(first, reset_first(src, subj, "toy-02"))


def reset_first(src, subj, scene):
    subj.reset()
    return subj.predict(src.load(src.frames(scene)[0]), ["memory"]).taps["memory"]


@check("Runner: a failing loader thread raises in the consumer; breaking out of the loop does not hang")
def _():
    from evalsuite.runner import Runner

    class Bad(toy.ToySource):
        def load(self, ref):
            if ref.index == 3:
                raise RuntimeError("disk gone")
            return super().load(ref)
    try:
        for _ in Runner(Bad(), toy.ToySubject(), "carry").iter_scene("toy-00"):
            pass
        raise AssertionError("must raise")
    except RuntimeError as e:
        assert "disk gone" in str(e)
    for i, _ in enumerate(Runner(toy.ToySource(), toy.ToySubject(), "carry").iter_scene("toy-00")):
        if i == 2:
            break


@check("Metrics: toy counts against a hand calculation; reduce = summed cells; scenes add up to the total")
def _():
    src, subj, task = toy.build()
    seg = task.metrics()[0]
    s = src.load(src.frames("toy-00")[3])
    p = subj.predict(s)
    c = seg.frame(s, p)["cnt"]
    pm, gm = p.outputs["mask"], s.reference["mask"]
    assert c[:, 0].sum() == (pm & gm).sum() and c[:, 1].sum() == (pm & ~gm).sum() and c[:, 2].sum() == (~pm & gm).sum()
    assert c.sum() == 48 * 48
    near = (pm[:, :24] & gm[:, :24]).sum() / ((pm[:, :24] | gm[:, :24]).sum())
    assert abs(seg.reduce([{"cnt": c}])["iou_near"] - near) < 1e-12
    all_counts = []
    for sc in src.scenes():
        subj.reset()
        all_counts.append([seg.frame(src.load(r), subj.predict(src.load(r))) for r in src.frames(sc)])
    tot = seg.reduce([c for cs in all_counts for c in cs])
    assert abs(tot["iou"] - direct_seg(src, subj)) < 1e-12
    assert np.allclose(sum(c["cnt"] for cs in all_counts for c in cs), sum(sum(c["cnt"] for c in cs) for cs in all_counts))
    assert np.isnan(seg.reduce([])["iou"])


# ------------------------------------------------------------------------------------------------ quantitative suite
@check("quant.run + table: the suite's numbers = the numbers computed directly without the suite")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        tab = suite.quant.table(st)
        assert len(st.frames) == 96 and st.manifest["finished"] and st.manifest["state"] == "carry"
        assert abs(tab["seg.iou"] - direct_seg(suite.source, suite.subject)) < 1e-12
        sc = suite.quant.per_scene(st)
        assert abs(sc["toy-03"]["seg.iou"] - direct_seg(suite.source, suite.subject, ["toy-03"])) < 1e-12
        st2 = suite.quant.run(name="b", progress=0)                                     # reproducible
        assert suite.quant.table(st2) == tab
        reset = suite.quant.run(name="c", state="reset", progress=0)
        assert abs(suite.quant.table(reset)["seg.iou"] - direct_seg(suite.source, suite.subject, reset=True)) < 1e-12
        assert suite.quant.table(reset)["seg.iou"] != tab["seg.iou"], "carry and reset must differ for a stateful network"


@check("quant.run: subsets (scenes, n_scenes, max_frames, frames), carry refuses a gap, resume completes an interrupted run")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        a = suite.quant.run(scenes=["toy-01", "toy-02"], max_frames=5, name="a", progress=0)
        assert len(a.frames) == 10 and a.scene_names() == ["toy-01", "toy-02"]
        b = suite.quant.run(n_scenes=3, seed=1, name="b", progress=0)
        assert len(b.scene_names()) == 3 and b.scene_names() == Sampler(suite.source, 1).scenes(3)
        gap = [f for f in Sampler(suite.source, 0).spread(10)]
        try:
            suite.quant.run(frames=gap, name="g", progress=0)
            raise AssertionError("carry with gaps must raise")
        except ValueError:
            pass
        r = suite.quant.run(frames=gap, state="reset", name="g", progress=0)
        assert {f.label for f in gap} == set(r.labels())
        one, two = [FrameRef("toy-01", 0)], [FrameRef("toy-01", 3)]         # another frame list under the same name is another run
        suite.quant.run(frames=one, state="reset", name="pick", progress=0)
        assert len(suite.quant.run(frames=one, state="reset", name="pick", progress=0).frames) == 1, "the same frames resume"
        try:
            suite.quant.run(frames=two, state="reset", name="pick", progress=0)
            raise AssertionError("another frame selection under the same run name must be refused")
        except ValueError as e:
            assert "frames_sha" in str(e), e
        import contextlib
        import io
        import json
        mf = os.path.join(suite.quant.dir, "pick", "manifest.json")       # a run stored before the frame list was part of the identity: resumes, with a warning
        old = json.load(open(mf))
        old["selection"].pop("frames_sha")
        json.dump(old, open(mf, "w"))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            suite.quant.run(frames=one, state="reset", name="pick", progress=0)
        assert "frames_sha" in err.getvalue() and "not verified" in err.getvalue(), err.getvalue()
        # interrupted run: a failing source in the middle; the resumed run gives the same numbers as an undisturbed one
        full = suite.quant.table(suite.quant.run(name="full", progress=0))

        class Flaky(toy.ToySource):
            boom = True

            def load(self, ref):
                if Flaky.boom and ref.scene == "toy-05" and ref.index == 4:
                    raise RuntimeError("crash")
                return super().load(ref)
        src, subj, task = Flaky(), toy.ToySubject(), toy.ToyTask()
        s2 = EvalSuite(src, subj, task, out=d)
        try:
            s2.quant.run(name="resumed", progress=0)
            raise AssertionError("must crash")
        except RuntimeError:
            pass
        part = RunStore.open(os.path.join(d, "quant", "resumed"))
        assert not part.manifest["finished"] and 0 < len(part.frames) < 96 and len(part.frames) % 1 == 0
        Flaky.boom = False
        done = s2.quant.run(name="resumed", progress=0)
        assert len(done.frames) == 96 and len(set(done.labels())) == 96 and done.manifest["finished"]
        assert s2.quant.table(done) == full, "a resumed run must give the numbers of an undisturbed run"


@check("state 'warmup': any subset of frames, each from an empty start with the network's warm-up history only; trace replays it exactly")
def _():
    from evalsuite.runner import Runner
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        sm = Sampler(suite.source, 0)
        frames = sm.spread(20, min_index=2)
        assert all(f.index >= 2 for f in frames) and len(frames) == 20
        wu = suite.quant.run(frames=frames, state="warmup", name="w", progress=0)
        rs = suite.quant.run(frames=frames, state="reset", name="r", progress=0)
        ca = suite.quant.run(scenes=sorted({f.scene for f in frames}), state="carry", name="c", progress=0)
        tw, tr = suite.quant.table(wu), suite.quant.table(rs)
        assert tw != tr, "the warm-up history changes the numbers of a stateful network"
        # each frame equals run_to with the network's warm-up history (2 frames) - and differs from the empty start
        runner = Runner(suite.source, suite.subject, "carry")
        for f in frames[:6]:
            s, p = runner.run_to(f)
            c = suite.task.metrics()[0].frame(s, p)["cnt"]
            assert np.array_equal(c, wu.counts("seg", [f.label])[0]["cnt"]), f.label
        tr_ = suite.quant.trace(wu, frames[3].label, ("overview",))
        assert abs(tr_.scores["seg.iou"] - wu.frame(frames[3].label)["scores"]["seg.iou"]) < 1e-12
        # the warm-up numbers lie between the empty start and the full-scene memory on average
        full = {r["label"]: r["scores"]["seg.iou"] for r in ca.frames}
        d_w = np.mean([abs(wu.frame(f.label)["scores"]["seg.iou"] - full[f.label]) for f in frames])
        d_r = np.mean([abs(rs.frame(f.label)["scores"]["seg.iou"] - full[f.label]) for f in frames])
        assert d_w <= d_r + 1e-12, (d_w, d_r)
        last = Sampler(suite.source, 0).spread(5, min_index=11)
        assert len(last) == 5 and all(f.index == 11 for f in last)


@check("outliers: the planted frames are found first; scene drag = leave-one-out by hand; frame drag is positive for them")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        osr = suite.quant.outliers(st, k=5)
        planted = {f"{s}/{i:03d}" for (s, i) in toy.PLANTED}
        iou = [r for r in osr["frames"] if r["metric"] == "seg.iou"]
        assert planted <= {r["label"] for r in iou[:5]}, [r["label"] for r in iou]
        assert all(r["outlier"] for r in iou if r["label"] in planted), "planted frames must exceed the z threshold"
        assert all(r["drag"]["iou"] > 0 for r in iou if r["label"] in planted)
        sc = [r for r in osr["scenes"] if r["metric"] == "seg.iou"]
        assert sc[0]["scene"] == "toy-03" and "toy-07" in [r["scene"] for r in sc[:5]], [r["scene"] for r in sc]
        assert all(r["drag"] > 0 for r in sc[:2]) and [r["drag"] for r in sc] == sorted((r["drag"] for r in sc), reverse=True)
        m = suite.task.metrics()[0]
        counts = st.counts("seg")
        full = m.reduce(counts)["iou"]
        wo = m.reduce([c for fr, c in zip(st.frames, counts) if fr["scene"] != "toy-03"])["iou"]
        assert abs(sc[0]["drag"] - (wo - full)) < 1e-12 and sc[0]["drag"] > 0
        # the 'low is better' metric: the same scenes also drag the centre error up
        oc = [r for r in osr["scenes"] if r["metric"] == "objs.center_err"]
        assert oc and oc[0]["drag"] > 0
        assert osr["flags"] == [] or all("status" in f for f in osr["flags"])
        assert all("qual frame" in r["view"] for r in osr["frames"])


@check("outliers: min_support keeps frames with nothing to find out of the ranking")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        every = suite.quant.outliers(st, k=200, min_support=0)["frames"]
        strict = suite.quant.outliers(st, k=200, min_support=10 ** 9)["frames"]
        assert len([r for r in every if r["metric"] == "seg.iou"]) > 0 and strict == []


@check("compare: a worse network is detected (b - a < 0, interval below 0); a run against itself gives exactly 0")
def _():
    with tempfile.TemporaryDirectory() as d:
        make(d).quant.run(name="a", progress=0)
        make(d, bias=0.35, name="worse").quant.run(name="b", progress=0)
        suite = make(d)
        cmp = suite.quant.compare("a", "b", boot=300)
        iou = cmp["metrics"]["seg.iou"]
        assert iou["diff"] < -0.02 and iou["hi"] < 0 and iou["p_b_better"] == 0.0 and iou["n_valid"] == 300, iou
        assert cmp["n_common_frames"] == 96
        # S8: runs of another source (here another world seed = other ground truth on the same frame labels) are refused
        other = make(d, seed=1).quant.run(name="o", progress=0)
        for fn in (lambda **kw: suite.quant.compare("a", "o", boot=20, **kw), lambda **kw: suite.quant.pareto(["a", "o"], **kw)):
            try:
                fn()
                raise AssertionError("runs of another source must be refused")
            except ValueError as e:
                assert "same source and task" in str(e) and "seed" in str(e), e
            fn(force=True)                                                  # force compares anyway
        assert other.manifest["source"]["seed"] == 1
        same = suite.quant.compare("a", "a", boot=100)["metrics"]["seg.iou"]
        assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
        make(d).quant.run(name="r", state="reset", progress=0)
        cm = suite.quant.compare("r", "a", boot=100)
        assert cm["n_common_frames"] == 96 and abs(cm["metrics"]["seg.iou"]["diff"]) > 0
        make(d).quant.run(scenes=["toy-00", "toy-01"], name="p", progress=0)
        assert suite.quant.compare("p", "a", boot=50)["n_common_frames"] == 24, "compare uses the frames both runs have"


@check("quant.report / trace / export_csv: files exist, every link resolves, the worst frame is drawn")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        path = suite.quant.report(st, render=3)
        links = report.links_in(path)
        assert links and all(os.path.exists(os.path.join(os.path.dirname(path), l)) for l in links), links
        txt = open(path, encoding="utf-8").read()
        assert "toy-03" in txt and "headline numbers" in txt and "qual frame" in txt
        csvp = os.path.join(st.path, "frames.csv")
        rows = open(csvp).read().strip().split("\n")
        assert len(rows) == 97 and "seg.iou" in rows[0]
        q = suite.quant.trace(st, "toy-07/002")
        assert os.path.exists(q.images["overview"]) and os.path.exists(q.images["debug"]) and q.taps_npz.endswith("_taps.npz")
        t = suite.quant.timing(n=3)
        assert t["median_ms"] > 0 and t["frame"] == "toy-00/000" and os.path.exists(os.path.join(d, "quant", "timing.json"))


# ------------------------------------------------------------------------------------------------ qualitative suite
@check("qual.frame / debug: files, header carries the label and the frame's own scores, taps dump equals the taps")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        r = suite.qual.debug("toy-03/5")
        assert os.path.exists(r.images["overview"]) and os.path.exists(r.images["debug"]) and os.path.exists(r.taps_npz)
        z = np.load(r.taps_npz)
        assert set(z.files) == {"blur", "memory", "logits", "features"}
        from evalsuite.runner import Runner
        sample, pred = Runner(suite.source, suite.subject, "carry").run_to(suite.source.find("toy-03/5"),
                                                                           ["blur", "memory", "logits", "features"])
        assert all(np.allclose(z[k], pred.taps[k]) for k in z.files)
        ctx = suite.qual.context(sample, pred)
        page = suite.qual.page(sample, pred, "overview", ctx)
        assert "toy-03/005" in page.title and "seg.iou" in page.title and "PLANTED: noise burst" in page.title
        # the score in the picture's header is the quantitative suite's score of this very frame: a trace replays the run
        # exactly (history 'full'); a quick look with the network's warm-up history (2 frames) is close but not identical
        st = suite.quant.run(scenes=["toy-03"], name="a", progress=0)
        tr = suite.quant.trace(st, "toy-03/005", ("overview",))
        assert abs(st.frame("toy-03/005")["scores"]["seg.iou"] - tr.scores["seg.iou"]) < 1e-12
        quick = suite.qual.frame("toy-03/5", history=None)
        assert abs(quick.scores["seg.iou"] - tr.scores["seg.iou"]) < 0.15 and quick.scores != tr.scores
        none = suite.qual.frame("toy-03/5", history="none")
        reset_run = make(d, state="reset").quant.run(scenes=["toy-03"], name="r", progress=0)
        assert abs(none.scores["seg.iou"] - reset_run.frame("toy-03/005")["scores"]["seg.iou"]) < 1e-12
        only = suite.qual.frame("toy-01/2", views=("debug",), taps=["blur"])
        assert set(np.load(only.taps_npz).files) == {"blur"}
        try:
            suite.qual.frame("toy-01/2", taps=["nope"])
            raise AssertionError("must raise")
        except KeyError:
            pass


@check("qual.random: n distinct frames, the video-first law's draws are reproducible, index.html links resolve")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        a = suite.qual.random(5, seed=9)
        b = suite.qual.random(5, seed=9)
        assert [r.ref for r in a] == [r.ref for r in b] == [f.label for f in Sampler(suite.source, 9).frames(5)]
        assert len({r.ref for r in a}) == 5 and all(os.path.exists(r.images["overview"]) for r in a)
        idx = os.path.join(d, "qualitative", "random_s9_n5", "index.html")
        links = report.links_in(idx)
        assert len(links) >= 10 and all(os.path.exists(os.path.join(os.path.dirname(idx), l)) for l in links)
        import json
        assert json.load(open(os.path.join(os.path.dirname(idx), "draws.json")))["frames"] == [r.ref for r in a]


@check("qual.videos: an H.264 (or gif) video with one picture per frame, read back; score strip and index exist")
def _():
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        out = suite.qual.videos(2, seed=3, fps=4)
        assert len(out) == 2 and out[0]["scene"] != out[1]["scene"]
        for o in out:
            v = o["video"]
            assert os.path.exists(v["path"]) and v["frames"] == 12, v
            assert v["codec"] in ("h264", "gif") and os.path.exists(o["scores_strip"])
            if v["codec"] == "h264":                                            # H.264 needs an even size (the gif fallback does not)
                assert v["size"][0] % 2 == 0 and v["size"][1] % 2 == 0, v
                video.verify(v["path"], 12)
        part = suite.qual.videos(1, scenes=["toy-04"], max_frames=5, render_every=2)[0]
        assert part["video"]["frames"] == 3, "render_every 2 of 5 frames = 3 pictures"
        idx = os.path.join(d, "qualitative", "videos", "index.html")
        assert all(os.path.exists(os.path.join(os.path.dirname(idx), l)) for l in report.links_in(idx))


# ------------------------------------------------------------------------------------------------ rendering
@check("figure: error colours at the right cells (TP grey, FP red, FN blue, not evaluated cream); counts")
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


@check("figure: every panel kind and overlay draws; page size follows the grid; PNG and array agree")
def _():
    from evalsuite import Shape
    a = np.random.default_rng(0).random((20, 30))
    shapes = [Shape("polygon", [(2, 2), (8, 2), (8, 8)], "r"), Shape("line", [(0, 0), (10, 10)]), Shape("points", [(5, 5), (6, 7)]),
              Shape("circle", (10, 10, 3)), Shape("arrow", (1, 1, 6, 6)), Shape("text", (5, 5), label="x")]
    panels = [Panel("image", "i", (a * 255).astype(np.uint8)), Panel("raster", "r", a, extent=(0, 30, 0, 20), colorbar=True, shapes=shapes,
                                                                    xlim=(30, 0), note="what to look for"),
              Panel("rgb", "c", np.dstack([a, a, a]), extent=(0, 30, 0, 20)), Panel("text", "t", "hello\nworld"),
              Panel("table", "tab", [["a", "b"], [1, 2.5]]), Panel("bars", "b", {"x": 1.0, "y": 2.0})]
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "p.png")
        img = figure.render_page(Page("title\nsecond line", panels, ncols=3, legend=figure.ERROR_LEGEND, footer="foot"), pth, dpi=60)
        from PIL import Image
        assert Image.open(pth).size == (img.shape[1], img.shape[0])
        assert abs(img.shape[1] / 60 - 3 * 4.6) < 0.1 and img.shape[0] / 60 > 2 * 4.2
        try:
            figure.render_page(Page("x", [Panel("nope")]))
            raise AssertionError("must raise")
        except ValueError:
            pass


@check("taps: energy, PCA, panels per kind, statistics, plausibility flags (dead, NaN, constant, no structure)")
def _():
    a = np.ones((3, 4, 5), np.float32)
    assert np.allclose(tapmod.energy(a), np.sqrt(3)) and tapmod.pca_rgb(np.random.default_rng(0).random((5, 8, 8))).shape == (8, 8, 3)
    c = tapmod.pca_rgb(np.random.default_rng(1).random((5, 8, 8)))
    assert c.min() >= 0 and c.max() <= 1
    specs = [TapSpec("fm", "feature_map", "s1"), TapSpec("lg", "logits"), TapSpec("im", "image"), TapSpec("mp", "map"),
             TapSpec("vec", "vector"), TapSpec("sc", "scalar")]
    taps = {"fm": np.random.default_rng(0).random((4, 6, 6)), "lg": np.random.default_rng(0).normal(size=(2, 6, 6)),
            "im": np.random.default_rng(0).random((6, 6, 3)), "mp": np.random.default_rng(0).random((6, 6)),
            "vec": np.arange(5.0), "sc": np.array(3.0)}
    kinds = [p.kind for p in tapmod.tap_panels(taps, specs)]
    assert kinds == ["raster", "rgb", "raster", "image", "raster", "bars", "text"], kinds
    assert [p.title for p in tapmod.tap_panels(taps, specs, wanted=["mp"])] == ["mp"]
    flags = lambda arr: {f.name.split(": ")[1]: f.status for f in tapmod.plausibility(TapSpec("t"), arr)}       # noqa: E731
    assert flags(np.zeros((2, 4, 4))) == {"alive": "fail"}
    assert flags(np.full((2, 4, 4), 3.0)) .get("alive") == "warn"
    nan = np.ones((2, 4, 4))
    nan[0, 0, 0] = np.nan
    assert flags(nan)["finite"] == "fail"
    assert flags(np.random.default_rng(0).random((2, 4, 4))) == {}
    assert flags(np.full((2, 4, 4), 1e9) + np.random.default_rng(0).random((2, 4, 4)))["magnitude"] == "warn"
    t = tapmod.stats_table({"a": a}, [Flag("a: alive", "fail", 0, "dead")])
    assert t.kind == "table" and any("FAIL" in str(r) for r in t.data)


@check("video: frames (arrays and files) -> mp4 read back with the right count and an even size; no frames raise")
def _():
    with tempfile.TemporaryDirectory() as d:
        frames = [np.full((75, 101, 3), 30 * i + 20, np.uint8) for i in range(5)]
        info = video.encode(frames, os.path.join(d, "v.mp4"), fps=2)
        assert info["frames"] == 5 and (info["codec"] == "gif" or (info["size"][0] % 2 == 0 and info["size"][1] % 2 == 0)), info
        try:
            video.encode([], os.path.join(d, "x.mp4"))
            raise AssertionError("must raise")
        except ValueError:
            pass
    with open(video.__file__, encoding="utf-8") as f:                       # where ffmpeg is looked up: no path of one machine
        assert "/mnt/c" not in f.read()
    old = os.environ.get("FFMPEG")
    try:
        os.environ["FFMPEG"] = "/no/such/ffmpeg"
        assert next(video._candidates()) == "/no/such/ffmpeg", "$FFMPEG is looked at first"
    finally:
        os.environ.pop("FFMPEG") if old is None else os.environ.__setitem__("FFMPEG", old)
    assert video._wsl_exe("ffmpeg.exe") == (os.name == "posix") and not video._wsl_exe("/usr/bin/ffmpeg"), "no wslpath under native Windows"


@check("report: tables, images and links; links_in finds every target")
def _():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "wb").write(b"x")
        p = report.write(os.path.join(d, "sub", "i.html"), "t <&>", [("h", report.table([["k", "v"], ["a", 1.5], ["w", "warn"]], status_col=1)),
                                                                  ("", report.image(os.path.join(d, "a.png"), os.path.join(d, "sub"), 100)),
                                                                  ("", report.link(os.path.join(d, "a.png"), os.path.join(d, "sub"), "x"))])
        txt = open(p, encoding="utf-8").read()
        assert "t &lt;&amp;&gt;" in txt and 'class="warn"' in txt and report.links_in(p) == ["../a.png"] * 3          # image: href + src, link: href


@check("torch_taps: hooks by module path (out / in / first / list / pick), arming, optional missing module, batch index")
def _():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("      (torch not installed: skipped)")
        return
    from evalsuite.torch_taps import Hook, TapRecorder

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
    assert got["a_list"].shape == (2, 4) and got["b_pick"].shape == (1,)
    assert rec.read(None)["a_out"].shape == (2, 4)
    try:
        rec.arm(["nope"])
        raise AssertionError("must raise")
    except KeyError:
        pass
    rec.close()
    m(x)


# ------------------------------------------------------------------------------------------------ probing
@check("probing: the budget adds up to the error cells counted directly; the place to work comes from Task.causes() (the cause name when it gives none); situations find the planted frames; sample / scan / CLI; an empty run has no situations")
def _():
    import io
    import contextlib
    from evalsuite.cli import main
    with tempfile.TemporaryDirectory() as d:
        suite = make(d)
        st = suite.quant.run(name="a", progress=0)
        src, subj = suite.source, suite.subject
        want = {"fn.near": 0.0, "fn.far": 0.0, "fp.near": 0.0, "fp.far": 0.0}
        for sc in src.scenes():                                       # the error cells without the suite (carry: reset per scene)
            subj.reset()
            for ref in src.frames(sc):
                s = src.load(ref)
                p, g = subj.predict(s).outputs["mask"], s.reference["mask"]
                for half, sl in (("near", slice(0, toy.SIZE // 2)), ("far", slice(toy.SIZE // 2, None))):
                    want[f"fn.{half}"] += float((~p & g)[:, sl].sum())
                    want[f"fp.{half}"] += float((p & ~g)[:, sl].sum())
        bud = {b["cause"]: b for b in suite.probe.budget(st)}
        assert all(abs(bud[c]["amount"] - want[c]) < 1e-6 for c in want), (bud, want)
        assert abs(sum(b["share"] for b in bud.values()) - 1) < 1e-12
        assert bud["fp.near"]["target"] == "code" and bud["fn.near"]["target"] == "model" and bud["fn.far"]["target"] == "fn.far"
        tg, sh = suite.probe.targets(list(bud.values())), {c: b["share"] for c, b in bud.items()}
        assert abs(tg["evaluation"] - sh["fp.far"] / 2) < 1e-12 and abs(tg["model"] - sh["fn.near"] - sh["fp.far"] / 2) < 1e-12 and abs(tg["fn.far"] - sh["fn.far"]) < 1e-12
        sits = {s["name"]: s for s in suite.probe.situations(st, min_frames=1)}
        planted = {f"{s}/{i:03d}" for (s, i) in toy.PLANTED}
        assert set(sits["planted"]["labels"]) == planted and sits["planted"]["excess"] > 0, sits.get("planted")
        smp = suite.probe.sample(st, "planted", 2, seed=1, render=False)
        assert len(smp) == 2 and {x["label"] for x in smp} <= planted
        res = suite.probe.scan(st, k=1, top=2, seed=0, render=False)
        assert os.path.exists(os.path.join(res["dir"], "budget.csv")) and os.path.exists(res["index"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["--adapter", "evalsuite.adapters.toy:build", "--arg", "n_scenes=3", "--out", d, "probe", "budget", "a"]) == 0
        assert "fp.near (code)" in buf.getvalue() and "fn.far (fn.far)" in buf.getvalue()
        empty = suite.quant.run(frames=[], name="e", progress=0)
        assert len(empty.frames) == 0 and suite.probe.situations("e") == [] and suite.probe.budget("e") == []


# ------------------------------------------------------------------------------------------------ the contract checker and the CLI
@check("check_adapter: a good adapter passes; broken adapters are caught with the reason")
def _():
    with tempfile.TemporaryDirectory() as d:
        good = check_adapter(make(d))
        assert all(r["ok"] for r in good) and len(good) >= 12

        def failed(suite):
            return [r["name"] for r in check_adapter(suite) if not r["ok"]]

        class WrongIdx(toy.ToySource):
            def frames(self, scene): return super().frames(scene)[1:]
        assert "source: scenes and frames" in failed(EvalSuite(WrongIdx(), toy.ToySubject(), toy.ToyTask(), out=d))

        class Undeclared(toy.ToySubject):
            def predict(self, sample, taps=()):
                p = super().predict(sample, taps)
                p.taps["extra"] = np.zeros(3)
                return p
        assert "subject: predict and taps" in failed(EvalSuite(toy.ToySource(), Undeclared(), toy.ToyTask(), out=d))

        class Stateful(toy.ToySubject):
            def reset(self): pass
        assert "subject: reset() restores the start state" in failed(EvalSuite(toy.ToySource(), Stateful(), toy.ToyTask(), out=d))

        from evalsuite.adapters import minimal                         # S6: a Subject WITHOUT taps whose reset() forgets nothing

        class LeakyNoTaps(minimal.MinSubject):
            def __init__(self):
                super().__init__()
                self.mem = None

            def predict(self, sample, taps=()):
                x = np.asarray(sample.inputs, float)
                self.mem = x if self.mem is None else 0.5 * self.mem + 0.5 * x          # a memory that reset() never clears
                return Prediction(outputs=self.mem > self.threshold)
        assert "subject: reset() restores the start state" in failed(EvalSuite(minimal.MinSource(), LeakyNoTaps(), minimal.MinTask(), out=d))
        assert not failed(EvalSuite(*minimal.build(), out=d)), "the stateless minimal adapter passes the reset check through its outputs"

        class OrderDep(toy.SegMetric):
            def reduce(self, counts, keys=None):
                r = super().reduce(counts)
                r["iou"] = float(counts[0]["cnt"][0, 0]) if counts else float("nan")      # depends on which frame is first
                return r

        class T2(toy.ToyTask):
            def metrics(self): return [OrderDep()]
        assert "metric seg: counts, reduce, scores" in failed(EvalSuite(toy.ToySource(), toy.ToySubject(), T2(), out=d))

        class NoOverview(toy.ToyTask):
            def views(self): return {}
        assert "task: has an overview view" in failed(EvalSuite(toy.ToySource(), toy.ToySubject(), NoOverview(), out=d))


@check("CLI: list, qual frame / debug / random / videos, quant run / table / per-scene / outliers / compare / trace / report / timing")
def _():
    import io
    import contextlib
    from evalsuite.cli import main
    with tempfile.TemporaryDirectory() as d:
        base = ["--adapter", "evalsuite.adapters.toy:build", "--out", d]
        calls = [["list"], ["list", "toy-02"], ["qual", "frame", "toy-03/5", "--view", "overview", "debug"], ["qual", "debug", "toy-01/1"],
                 ["qual", "random", "3", "--seed", "2"], ["qual", "videos", "1", "--seed", "1", "--max-frames", "4"],
                 ["quant", "run", "--name", "a", "--n-scenes", "4"], ["quant", "table", "a"], ["quant", "per-scene", "a"],
                 ["quant", "outliers", "a", "--k", "3"], ["quant", "trace", "a", "@FIRST"], ["quant", "report", "a", "--render", "2"],
                 ["quant", "timing", "--n", "3"]]
        for c in calls:
            if "@FIRST" in c:                                                  # a frame that the random 4-scene run really has
                c = [RunStore.open(os.path.join(d, "quant", "a")).frames[1]["label"] if x == "@FIRST" else x for x in c]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(base + c)
            assert rc == 0 and buf.getvalue().strip(), c
        with contextlib.redirect_stdout(io.StringIO()):
            main(base + ["--arg", "bias=0.35", "--arg", "name=worse", "quant", "run", "--name", "b", "--n-scenes", "4"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(base + ["quant", "compare", "a", "b", "--boot", "50"])
        assert '"seg.iou"' in buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(base + ["selftest-adapter"]) == 0
        assert "checks passed" in buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(base + ["quant", "run", "--name", "s", "--spread", "20"])
        assert "20 frames" in buf.getvalue()


@check("minimal adapter: the smallest adapter (frame + reduce + one view) runs the core commands, also as a copy anywhere; default score = the frame's own IoU; probing loads lazily")
def _():
    import io
    import contextlib
    import json
    import shutil
    from evalsuite.adapters import minimal
    from evalsuite.cli import main
    src, subj, m = minimal.MinSource(), minimal.MinSubject(), minimal.IoU()
    s = src.load(src.frames("demo-0")[2])
    c = m.frame(s, subj.predict(s))
    assert m.score(c) == {"iou": m.reduce([c])["iou"]} and m.support(c) == {}, "default score = reduce of the frame alone, no support"
    assert m.score({"c": [3, 1, 1, 0]}) == {"iou": 0.6}, "counts given as plain lists work in the default score too"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(minimal.__file__, os.path.join(d, "my_adapter.py"))                   # the recipe: copy the file, point --adapter at it
        for spec in ("evalsuite.adapters.minimal:build", os.path.join(d, "my_adapter.py") + ":build"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert main(["--adapter", spec, "--out", d, "list"]) == 0
            assert "demo-1 6" in buf.getvalue(), (spec, buf.getvalue())
        base = ["--adapter", "evalsuite.adapters.minimal:build", "--out", d]

        def call(*args):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(base + list(args))
            assert rc == 0 and buf.getvalue().strip(), args
            return buf.getvalue()
        call("qual", "frame", "demo-1/3")
        call("qual", "random", "2", "--seed", "1")
        call("qual", "videos", "1", "--max-frames", "3")
        assert "iou.iou" in call("quant", "run", "--name", "a")
        call("quant", "table", "a")
        call("quant", "per-scene", "a")
        call("quant", "report", "a", "--render", "1")
        call("quant", "trace", "a", "demo-1/3")                               # the unpadded label (stored as demo-1/003)
        call("quant", "timing", "--n", "3", "--run", "a")
        assert "checks passed" in call("selftest-adapter")
        call("--arg", "threshold=0.1", "quant", "run", "--name", "b")
        assert '"iou.iou"' in call("quant", "compare", "a", "b", "--boot", "50")
        st = RunStore.open(os.path.join(d, "quant", "a"))
        assert len(st.frames) == 18
        p, g = subj.predict(s).outputs, s.reference
        want = (p & g).sum() / (p | g).sum()                                  # intersection over union, written out here
        assert abs(st.frame("demo-0/002")["scores"]["iou.iou"] - want) < 1e-12, "the stored frame score is the frame's own IoU, computed independently"
        scores = [fr["scores"]["iou.iou"] for fr in st.frames]
        assert len({round(v, 6) for v in scores}) > 3 and all(0 <= v <= 1 for v in scores), "the default scores differ from frame to frame"
        out = json.loads(call("quant", "outliers", "a", "--k", "2"))
        assert out["frames"] and all(r["label"] in st.labels() for r in out["frames"]), "the outlier search needs no score() of the adapter"
        try:                                                                  # a resume with another threshold is another run: refused
            call("--arg", "threshold=0.9", "quant", "run", "--name", "a")
            raise AssertionError("a resume with another threshold must be refused")
        except ValueError as e:
            assert "different run" in str(e) and "threshold" in str(e)
        suite = EvalSuite(*minimal.build(), out=d)                            # probing is loaded when it is first used
        assert suite._probe is None
        p = suite.probe
        assert p is suite.probe and type(p).__name__ == "Prober"


def main(only=()) -> int:
    from evalsuite.selftest_cost import CHECKS as COST_CHECKS
    from evalsuite.selftest_comparison import CHECKS as COMPARISON_CHECKS
    checks = CHECKS + COST_CHECKS + COMPARISON_CHECKS                     # the lane Task's tests live in the lane project (D24)
    if only:                                                               # name filter: python -m evalsuite selftest compare
        checks = [c for c in checks if any(o in c[0] for o in only)]
    t0 = time.time()
    bad = 0
    for name, fn in checks:
        t = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t:.1f} s)")
        except Exception:                                                  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"evalsuite selftest: {len(checks) - bad}/{len(checks)} checks passed in {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

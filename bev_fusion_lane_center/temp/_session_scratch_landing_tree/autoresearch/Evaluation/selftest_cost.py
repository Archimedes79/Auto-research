"""Known-answer tests of the cost side of the suite (`cost.py`, `Subject.ops`, `torch_taps.count_ops`, the ms per frame of a run, `quant timing --run`,
`quant pareto`). Registered into the core self-test (`python -m Evaluation selftest cost`)."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
from typing import Callable, List, Tuple

import numpy as np

from . import EvalSuite
from .adapters import toy
from .cli import main as cli
from .cost import format_table, front, with_estimate

CHECKS: List[Tuple[str, Callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def run_toy(d: str, name: str, bias: float = 0.0, subject_cls=toy.ToySubject, **kw):
    suite = EvalSuite(toy.ToySource(n_scenes=4, n_frames=6), subject_cls(bias=bias, name=name), toy.ToyTask(), out=d)
    return suite, suite.quant.run(name=name, progress=0, **kw)


@check("cost.front: not beaten on both axes; ties stay, a missing value is out, lower-is-better flips the quality direction")
def _():
    nan = float("nan")
    #            cost: 1    2    3    4    2    5
    q = [0.5, 0.8, 0.7, 0.9, 0.8, 0.9]
    assert front([1, 2, 3, 4, 2, 5], q) == [True, True, False, True, True, False]       # (3, .7) beaten by (2, .8); (5, .9) by (4, .9): same quality, dearer
    assert front([1, 2, 3], [0.9, 0.5, 0.7], high=False) == [True, True, False], "lower is better: (2, .5) is not beaten, (3, .7) is beaten by (2, .5)"
    assert front([1, 2, nan], [0.5, 0.6, 0.9]) == [True, True, False] and front([1, 2], [0.5, nan]) == [True, False]
    assert front([1, 1], [0.5, 0.5]) == [True, True], "equal points do not beat each other"
    assert front([], []) == []


@check("cost.with_estimate: adds a part as an estimate with its formula; measured becomes mixed")
def _():
    a = with_estimate({"gmacs": 2.0, "method": "measured", "note": "convs", "parts": {"cnn": 2.0}}, "splat", 0.5, "H*W*K = 200*200*12 MAC")
    assert a["gmacs"] == 2.5 and a["method"] == "mixed" and a["parts"] == {"cnn": 2.0, "splat": 0.5}
    assert "splat: estimated, H*W*K = 200*200*12 MAC" in a["note"] and a["note"].startswith("convs")
    b = with_estimate(None, "x", 1.0, "by hand")
    assert b["gmacs"] == 1.0 and b["method"] == "estimated" and b["note"] == "x: estimated, by hand"
    assert with_estimate(b, "y", 2.0, "by hand")["method"] == "estimated"


@check("torch_taps.count_ops: a small network against the hand count (convolutions and linear layers; ReLU and flatten add nothing)")
def _():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("      (torch not installed: skipped)")
        return
    from .torch_taps import count_ops
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1), nn.Flatten(), nn.Linear(4 * 16 * 16, 10))
    x = torch.zeros(1, 3, 16, 16)
    r = count_ops(lambda: m(x))
    conv1, conv2, lin = 16 * 16 * 8 * 3 * 9, 16 * 16 * 4 * 8, 1024 * 10
    assert round(r["gmacs"] * 1e9) == conv1 + conv2 + lin, r
    assert r["method"] == "measured" and "not counted" in r["note"]
    assert round(r["parts"]["Sequential.0"] * 1e9) == conv1 and round(r["parts"]["Sequential.2"] * 1e9) == conv2 and round(r["parts"]["Sequential.4"] * 1e9) == lin
    assert "Global" not in r["parts"] and "Sequential" not in r["parts"], "the root and the total are not parts"
    again = count_ops(lambda: (m(x), m(x)))                                       # two calls are counted twice
    assert round(again["gmacs"] * 1e9) == 2 * (conv1 + conv2 + lin)

    class Deep(nn.Module):                                 # four levels: depth limits the parts to the levels below the root
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Sequential(nn.Linear(8, 8)))
            self.b = nn.Linear(8, 8)

        def forward(self, z):
            return self.b(self.a(z))
    deep = Deep()
    p1, p3 = count_ops(lambda: deep(torch.zeros(1, 8)), depth=1)["parts"], count_ops(lambda: deep(torch.zeros(1, 8)), depth=3)["parts"]
    assert set(p1) == {"Deep.a", "Deep.b"}, p1
    assert set(p3) == {"Deep.a", "Deep.a.0", "Deep.a.0.0", "Deep.b"}, p3
    assert all(round(v * 1e9) == 64 for v in p3.values()), "a nested module is also counted in its parent"


@check("quant.run records ms per frame and Subject.ops; ops that touch the memory or fail change no number; the report and the benchmark (timing --run) show them")
def _():
    class Touchy(toy.ToySubject):                          # an ops() that runs the network (as a counter does) must not leak its state into the run
        def ops(self, sample):
            self.predict(sample)
            return super().ops(sample)

    class Failing(toy.ToySubject):
        def ops(self, sample):
            raise RuntimeError("no counter here")

    class NoOps(toy.ToySubject):
        ops = toy.Subject.ops
    with tempfile.TemporaryDirectory() as d:
        suite, st = run_toy(d, "plain")
        c = st.manifest["cost"]
        assert c["run_ms"]["n"] == 24 and c["run_ms"]["median_ms"] > 0 and c["run_ms"]["p90_ms"] >= c["run_ms"]["median_ms"], c
        assert c["ops"]["method"] == "estimated" and math.isclose(c["ops"]["gmacs"], 48 * 48 * 11 / 1e9) and "bench" not in c
        base = suite.quant.table(st)
        _, st2 = run_toy(d, "touchy", subject_cls=Touchy)
        assert suite.quant.table(st2) == base, "ops() may run the network: the suite resets the memory afterwards"
        _, st3 = run_toy(d, "failing", subject_cls=Failing)
        assert "ops" not in st3.manifest["cost"] and len(st3.frames) == 24 and suite.quant.table(st3) == base
        _, st4 = run_toy(d, "noops", subject_cls=NoOps)
        assert "ops" not in st4.manifest["cost"], "a Subject without ops() has no operations entry"
        res = suite.quant.timing(n=3, run="plain")
        assert suite.quant.open("plain").manifest["cost"]["bench"]["median_ms"] == res["median_ms"] > 0
        assert res["load"].get("host"), "the benchmark says where it was taken"
        html = open(suite.quant.report("plain", render=1), encoding="utf-8").read()
        assert "GMACs per frame" in html and "ms per frame, benchmark" in html and "ms per frame, measured in the run" in html
        again = suite.quant.run(name="plain", progress=0)                            # a finished run resumed: nothing is run, the entries stay
        assert again.manifest["cost"]["ops"]["gmacs"] == c["ops"]["gmacs"] and again.manifest["cost"]["bench"]["n"] == 3
        assert again.manifest["cost"]["run_ms"] == c["run_ms"], "no new frame: the run time is not replaced by the load of the resume"
        # a benchmark of another network is not attached to this run
        other = EvalSuite(toy.ToySource(n_scenes=4, n_frames=6), toy.ToySubject(bias=0.4, name="plain"), toy.ToyTask(), out=d)
        try:
            other.quant.timing(n=3, run="plain")
            raise AssertionError("a benchmark of another Subject must be refused")
        except ValueError as e:
            assert "bias" in str(e)
        assert other.quant.timing(n=3, run="plain", force=True)["n"] == 3
        # the reset after ops(): called directly, the memory the counter filled is gone
        sub = Touchy(name="x")
        ts = EvalSuite(toy.ToySource(n_scenes=4, n_frames=6), sub, toy.ToyTask(), out=d)
        assert sub.mem is None and ts.quant._ops(ts.source.frames("toy-00")[0], "x")["gmacs"] > 0 and sub.mem is None, "ops() runs the network, then the suite resets it"
        # an interrupted run resumed: the run time covers every stored frame, not only those of the last session

        class Dying(toy.ToySubject):
            calls = 0

            def predict(self, sample, taps=()):
                Dying.calls += 1
                if Dying.calls == 11:
                    raise RuntimeError("killed")
                return super().predict(sample, taps)
        dying = EvalSuite(toy.ToySource(n_scenes=4, n_frames=6), Dying(name="cut"), toy.ToyTask(), out=d)
        try:
            dying.quant.run(name="cut", progress=0)
            raise AssertionError("the planted failure must reach the caller")
        except RuntimeError:
            pass
        part = dying.quant.open("cut")
        assert not part.manifest["finished"] and 0 < len(part.frames) < 24 and part.manifest["cost"]["run_ms"]["n"] == len(part.frames), "frames are flushed before the run ends"
        _, full = run_toy(d, "cut")
        assert len(full.frames) == 24 and full.manifest["cost"]["run_ms"]["n"] == 24, full.manifest["cost"]
        # what ops() returns is made a plain JSON dict; anything else is not counted and does not stop the run

        class Numpy(toy.ToySubject):
            def ops(self, sample):
                return {"gmacs": np.float32(1.5), "method": "measured", "parts": {"a": np.float32(1.0), "b": np.array(0.5)}}

        class Bare(toy.ToySubject):
            def ops(self, sample):
                return 3.0
        _, stn = run_toy(d, "numpy", subject_cls=Numpy)
        on_disk = json.load(open(os.path.join(d, "quant", "numpy", "manifest.json")))["cost"]["ops"]
        assert on_disk["gmacs"] == 1.5 and on_disk["parts"] == {"a": 1.0, "b": 0.5}, on_disk
        assert "1.5" in open(suite.quant.report("numpy", render=0), encoding="utf-8").read(), "the report works with numpy numbers from ops()"
        _, stb = run_toy(d, "bare", subject_cls=Bare)
        assert "ops" not in stb.manifest["cost"] and len(stb.frames) == 24 and "GMACs" not in open(suite.quant.report("bare", render=0), encoding="utf-8").read()


@check("quant pareto: the front of planted costs (higher and lower is better), common frames, mixed ms sources, a run without ops, files, CLI")
def _():
    with tempfile.TemporaryDirectory() as d:
        biases = [0.0, 0.1, 0.2, 0.3]
        names = [f"b{int(b * 100):02d}" for b in biases]
        suite = None
        for b, n in zip(biases, names):
            suite, _ = run_toy(d, n, bias=b)
        q = suite.quant
        for y, high in (("seg.iou", True), ("objs.center_err", False)):
            val = {n: q.table(n)[y] for n in names}
            order = sorted(names, key=lambda n: -val[n] if high else val[n])             # best quality first
            assert len({round(v, 9) for v in val.values()}) == 4, val
            for n, ms in zip(order, (9.0, 5.0, 6.0, 3.0)):                                 # the third is slower than the second and worse: beaten
                q.open(n).set_cost("bench", {"median_ms": ms, "p90_ms": ms, "mean_ms": ms, "n": 5, "load": {"load1": 0.2}})
            res = q.pareto(names, y=y)
            on = {r["run"] for r in res["rows"] if r["front_ms"]}
            assert on == {order[0], order[1], order[3]}, (y, order, on)
            assert res["better"] == ("high" if high else "low") and res["ms_source"] == "bench" and res["n_common_frames"] == 24
            assert {r["run"] for r in res["rows"] if r["front_gmacs"]} == {order[0]}, "same GMACs everywhere: only the best quality is not beaten"
            assert os.path.getsize(res["png"]) > 1000 and json.load(open(res["json"]))["rows"][0]["run"] in names
            text = format_table(res)
            assert order[0] in text and "common frames" in text
        # a benchmark taken on a busy machine is flagged (load average above half the CPUs)
        busy = {"median_ms": 5.0, "p90_ms": 5.0, "mean_ms": 5.0, "n": 5, "load": {"load1": 9.0, "cpus": 8}}
        q.open(names[0]).set_cost("bench", busy)
        assert any("busy" in n and names[0] in n for n in q.pareto(names)["notes"])
        q.open(names[0]).set_cost("bench", {**busy, "load": {"load1": 3.0, "cpus": 8}})
        assert not any("busy" in n for n in q.pareto(names)["notes"])
        # a run on fewer frames: quality is taken on the common frames, and the table says so
        _, _ = run_toy(d, "few", bias=0.0, max_frames=3)
        q.open("few").set_cost("bench", {"median_ms": 4.0, "p90_ms": 4.0, "mean_ms": 4.0, "n": 5, "load": {"load1": 0.2}})
        res = q.pareto(names + ["few"])
        assert res["n_common_frames"] == 12 and any("different frames" in n for n in res["notes"])
        ref = q.table(names[0], labels=[fr["label"] for fr in q.open("few").frames])["seg.iou"]
        assert math.isclose(next(r["quality"] for r in res["rows"] if r["run"] == names[0]), ref), "quality on the common frames"
        # where the ms come from is said: no benchmark for one run -> the times measured during the runs, flagged; neither source for all -> mixed
        def drop(run, *keys):
            st = q.open(run)
            for k in keys:
                st.manifest["cost"].pop(k)
            st._write_manifest()
        drop("few", "bench")
        res = q.pareto(names + ["few"])
        assert res["ms_source"] == "run_ms" and any("not an idle benchmark" in n for n in res["notes"])
        drop(names[0], "run_ms")                                   # now b00 has only a benchmark and "few" only a run time
        res = q.pareto(names + ["few"])
        assert res["ms_source"] == "mixed" and any("not comparable" in n for n in res["notes"])
        # a run without operations is not on the gmacs front, and the table says so
        drop("few", "ops")
        res = q.pareto(names + ["few"], costs=("gmacs",))
        assert not next(r for r in res["rows"] if r["run"] == "few")["front_gmacs"] and any("no gmacs" in n for n in res["notes"])
        assert not any("runtime" in n or "ms " in n or "benchmark" in n for n in res["notes"]), "asking for gmacs only: no remarks about ms"
        # wrong questions get an answer, not a silent nan
        for bad in ("seg.nokey", "seg", "nometric.iou"):
            try:
                q.pareto(names, y=bad)
            except KeyError:
                pass
            else:
                raise AssertionError(f"the quality axis {bad!r} must be refused")
        # runs without any cost entry (an older suite): no ms, nothing to draw, and the table says why
        for n, b in (("nc1", 0.1), ("nc2", 0.2)):
            run_toy(d, n, bias=b)
            drop(n, "run_ms", "ops")
        res = q.pareto(["nc1", "nc2"])
        assert res["ms_source"] == "none" and res["png"] is None and any("no run has a runtime" in n for n in res["notes"]), res
        # ms taken on different machines are flagged; a path without an extension still gives a png
        for n, host in ((names[1], "box-a"), (names[2], "box-b")):
            q.open(n).set_cost("bench", {"median_ms": 5.0, "p90_ms": 5.0, "mean_ms": 5.0, "n": 5, "load": {"load1": 0.1, "cpus": 8, "host": host}})
        res = q.pareto(names[1:3], out=os.path.join(d, "plain_path"))
        assert any("different machines" in n for n in res["notes"]) and res["png"].endswith(".png") and os.path.exists(res["png"])
        # the command line (the adapter must describe the runs it timed: same Source, same Subject)
        arg = ["--arg", "n_scenes=4", "--arg", "n_frames=6"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli(["--adapter", "Evaluation.adapters.toy:build", "--out", d, "quant", "pareto", *names, "--y", "seg.iou"]) == 0
        assert "front" in buf.getvalue() and "picture:" in buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli(["--adapter", "Evaluation.adapters.toy:build", "--out", d, *arg, "--arg", f"name={names[0]}", "quant", "timing", "--n", "3", "--run", names[0]]) == 0
        assert q.open(names[0]).manifest["cost"]["bench"]["n"] == 3
        try:
            cli(["--adapter", "Evaluation.adapters.toy:build", "--out", d, *arg, "--arg", "name=other", "quant", "timing", "--n", "3", "--run", names[0]])
            raise AssertionError("another Subject on the command line must be refused")
        except ValueError as e:
            assert "name" in str(e)

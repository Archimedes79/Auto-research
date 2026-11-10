"""The contract checker: does an adapter (Source + Subject + Task) keep what the suite relies on? Runs on the first scene of the
Source, a handful of frames, and reports every check as ok / fail with the reason, so a new adapter is debugged in one command:

    python -m Evaluation --adapter pkg.module:build selftest-adapter

Checks: scenes and frames (non-empty, indices 0..n-1 in order, find() round trip by label / tuple / key), a frame loads with the
right ref, the Subject predicts a Prediction and its taps are the declared ones with finite numpy arrays, every Metric returns
Counts of numpy arrays whose reduce() works on one frame, on several, on none, whose headline keys are produced, whose scores are
numbers, order invariance (reduce of the same frames in another order gives the same numbers), every View renders a Page that draws, the Task has an 'overview', state handling (after reset() the same frame gives the same taps AND outputs, so a Subject without taps is checked too).
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List

import numpy as np

from .contracts import Page, Prediction, Sample
from .render import figure


def _run(results: List[Dict[str, Any]], name: str, fn: Callable[[], Any]) -> Any:
    try:
        note = fn()
        results.append({"name": name, "ok": True, "note": "" if note is None else str(note)})
    except Exception as e:                                              # noqa: BLE001
        results.append({"name": name, "ok": False, "note": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"})


def _differs(x: Any, y: Any, path: str = "outputs") -> Any:
    """None when two predictions agree, else where they differ. Arrays and numbers: within 1e-5 + 1e-3 of the scale (float noise of a
    GPU network grows with depth, 1e-4 of the scale; a leaked state is of the scale itself); dicts, lists, tuples and objects with
    attributes (dataclasses) are compared entry by entry; anything else with ==."""
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return f"{path}: keys {sorted(map(str, x))} vs {sorted(map(str, y))}"
        for k in x:
            bad = _differs(x[k], y[k], f"{path}[{k!r}]")
            if bad:
                return bad
        return None
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return f"{path}: length {len(x)} vs {len(y)}"
        for i, (u, v) in enumerate(zip(x, y)):
            bad = _differs(u, v, f"{path}[{i}]")
            if bad:
                return bad
        return None
    if hasattr(x, "detach") and hasattr(x, "cpu"):                   # a torch tensor (no torch import here): compare its values
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach") and hasattr(y, "cpu"):
        y = y.detach().cpu().numpy()
    if isinstance(x, (np.ndarray, np.generic, int, float, bool)) and isinstance(y, (np.ndarray, np.generic, int, float, bool)):
        a, b = np.asarray(x), np.asarray(y)
        if a.shape != b.shape:
            return f"{path}: shape {a.shape} vs {b.shape}"
        if a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
            return None if np.array_equal(a, b) else f"{path}: values differ"
        a, b = a.astype(np.float64), b.astype(np.float64)
        tol = 1e-5 + 1e-3 * (np.nanmax(np.abs(a)) if a.size and np.isfinite(a).any() else 0.0)
        if not np.allclose(a, b, atol=tol, rtol=0, equal_nan=True):
            return f"{path}: max difference {np.nanmax(np.abs(a - b)):.3g}"
        return None
    if hasattr(x, "__dict__") and type(x) is type(y) and not callable(x):
        return _differs(vars(x), vars(y), path)
    try:
        return None if bool(x == y) else f"{path}: {x!r} != {y!r}"[:200]
    except Exception:                                                   # noqa: BLE001  (not comparable)
        return None if type(x) is type(y) else f"{path}: {type(x).__name__} vs {type(y).__name__}"


def check_adapter(suite, n_frames: int = 4) -> List[Dict[str, Any]]:
    src, subj, task = suite.source, suite.subject, suite.task
    R: List[Dict[str, Any]] = []
    box: Dict[str, Any] = {}

    def scenes():
        sc = src.scenes()
        assert sc and len(set(sc)) == len(sc), "scenes() must be non-empty and unique"
        for s in sc[:3]:
            fr = src.frames(s)
            assert fr, f"scene {s} has no frames"
            assert [f.index for f in fr] == list(range(len(fr))), f"{s}: frame indices must be 0..n-1 in time order"
            assert all(f.scene == s for f in fr)
        box["scene"] = sc[0]
        box["frames"] = src.frames(sc[0])[:n_frames]
        return f"{len(sc)} scenes, first has {len(src.frames(sc[0]))} frames"
    _run(R, "source: scenes and frames", scenes)
    if not R[-1]["ok"]:
        return R

    def find():
        f = box["frames"][-1]
        assert src.find(f.label) == f and src.find((f.scene, f.index)) == f and src.find(f) == f
        if f.key:
            assert src.find(f.key) == f, "find(key) must return the frame"
        try:
            src.find("no-such-scene/1")
            raise AssertionError("find() of an unknown scene must raise KeyError")
        except KeyError:
            pass
    _run(R, "source: find by label / tuple / key", find)

    def load():
        s = src.load(box["frames"][0])
        assert isinstance(s, Sample) and s.ref == box["frames"][0], "load() must return a Sample with the same ref"
        box["sample"] = s
    _run(R, "source: load", load)
    if not R[-1]["ok"]:
        return R

    declared = {t.name: t for t in subj.taps()}

    def predict():
        subj.reset()
        p = subj.predict(box["sample"], list(declared))
        assert isinstance(p, Prediction), "predict() must return a Prediction"
        extra = set(p.taps) - set(declared)
        assert not extra, f"taps returned but not declared: {sorted(extra)}"
        missing = {n for n in set(declared) - set(p.taps) if not declared[n].optional}
        assert not missing, f"taps declared but not returned: {sorted(missing)} (mark a tap optional=True if the network may lack it)"
        for n, a in p.taps.items():
            assert isinstance(a, np.ndarray) and np.isfinite(a.astype(float)).all(), f"tap {n}: finite numpy array expected"
        return f"{len(p.taps)} taps"
    _run(R, "subject: predict and taps", predict)
    if not R[-1]["ok"]:
        return R

    def reset_same():
        f0 = box["sample"]
        f1 = src.load(box["frames"][1]) if len(box["frames"]) > 1 else f0
        subj.reset()
        a = subj.predict(f0, list(declared))
        subj.predict(f1, ())                                      # change the state ...
        subj.reset()                                              # ... and forget it
        b = subj.predict(f0, list(declared))
        for n in a.taps:
            bad = _differs(a.taps[n], b.taps[n])
            assert bad is None, f"after reset() the same frame gave another tap {n} ({bad})"
        bad = _differs(a.outputs, b.outputs)                       # a Subject without taps is checked through its outputs
        assert bad is None, f"after reset() the same frame gave other outputs ({bad})"
    _run(R, "subject: reset() restores the start state", reset_same)

    def warm():
        assert isinstance(subj.warmup, int) and subj.warmup >= 0
    _run(R, "subject: warmup is a non-negative int", warm)

    metrics = task.metrics()

    def names():
        ns = [m.name for m in metrics]
        assert ns and len(set(ns)) == len(ns), "metric names must be unique and there must be at least one metric"
        return ns
    _run(R, "task: metric names", names)
    for m in metrics:
        def one(m=m):
            counts = []
            subj.reset()
            for r in box["frames"]:
                s = src.load(r)
                p = subj.predict(s, ())
                c = m.frame(s, p)
                assert isinstance(c, dict) and all(isinstance(v, np.ndarray) or np.isscalar(v) for v in c.values()), \
                    "frame() must return a dict of numpy arrays"
                sc = m.score(c)
                assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in sc.values()), "score() values must be numbers"
                m.support(c)
                counts.append({k: np.asarray(v) for k, v in c.items()})
            full = m.reduce(counts)
            assert isinstance(full, dict) and all(isinstance(v, (int, float, np.integer, np.floating)) for v in full.values()), \
                "reduce() must return a flat dict of numbers"
            for h in m.headline:
                assert h in full, f"headline key {h!r} is not produced by reduce()"
            m.reduce(counts[:1])
            m.reduce([])
            rev = m.reduce(counts[::-1])
            for k, v in full.items():
                w = rev.get(k, float("nan"))
                assert (v != v and w != w) or abs(v - w) <= 1e-6 * max(1.0, abs(v)), f"reduce() depends on the frame order ({k})"
            if m.headline:
                part = m.reduce(counts, keys=list(m.headline))
                for h in m.headline:
                    a, b = part[h], full[h]
                    assert (a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"reduce(keys=) differs from reduce() for {h}"
            for k in m.better:
                assert m.better[k] in ("high", "low")
            return f"{len(full)} numbers"
        _run(R, f"metric {m.name}: counts, reduce, scores", one)

    views = task.views()

    def has_overview():
        assert "overview" in views, "the Task must offer an 'overview' view"
    _run(R, "task: has an overview view", has_overview)
    from .qualitative import DebugView
    allv = dict(views)
    allv.setdefault("debug", DebugView())
    for name, v in allv.items():
        def draw(name=name, v=v):
            subj.reset()
            need = [t.name for t in subj.taps()] if getattr(v, "needs_taps", False) else []
            s = box["sample"]
            p = subj.predict(s, need)
            ctx = suite.qual.context(s, p)
            page = suite.qual.page(s, p, name, ctx)
            assert isinstance(page, Page) and page.panels, "render() must return a Page with panels"
            img = figure.render_page(page)
            assert img.ndim == 3 and img.shape[2] == 3 and img.min() != img.max(), "the page drew a blank image"
            return f"{len(page.panels)} panels, image {img.shape[1]}x{img.shape[0]}"
        _run(R, f"view {name}: renders", draw)
    return R


def format_results(results: List[Dict[str, Any]]) -> str:
    lines = [f"{'ok  ' if r['ok'] else 'FAIL'} {r['name']}" + (f"  ({r['note']})" if r["ok"] and r["note"] else "") for r in results]
    bad = [r for r in results if not r["ok"]]
    for r in bad:
        lines.append(f"--- {r['name']}\n{r['note']}")
    lines.append(f"{len(results) - len(bad)}/{len(results)} checks passed")
    return "\n".join(lines)

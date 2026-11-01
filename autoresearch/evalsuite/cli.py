"""Command line of the suite.

    python -m evalsuite --adapter pkg.module:build [--arg key=value ...] [--out DIR] [--state carry|reset|warmup] <command>

    list [SCENE]                                         scenes (and the frames of one scene)
    qual frame SPEC [--view overview debug] [--taps a b] [--history warmup|full|none]   SPEC = scene-0061/12 | scene-0061:12 | a frame key
    qual debug SPEC [--taps a b] [--history ...]         the overview next to the network parts of one frame
    qual random N [--seed S] [--view ...]                N frames: a random video first, then a random frame in it
    qual videos M [--seed S] [--view V] [--fps F] [--max-frames N] [--scenes s1 s2]
    quant run [--n-scenes N | --scenes s1 s2] [--spread N] [--max-frames N] [--seed S] [--name NAME]
    quant table RUN | per-scene RUN | outliers RUN [--k K] [--z Z] | trace RUN LABEL (scene/3 = scene/003) | report RUN [--render K]
    quant compare CONTROL VARIANT [--boot N] [--seed S] [--force]   paired scene bootstrap, diff = variant - control; refused when the two
                                                         runs differ in source or task (GT version, cache) unless --force
    quant timing [--frame SPEC] [--n N] [--run RUN]
    quant pareto RUN [RUN ...] [--y metric.key] [--cost ms gmacs] [--force]   quality against runtime / operations, the runs not beaten on both
    selftest                                             the core against known answers on the toy adapter (no --adapter)
    selftest-adapter                                     the contract checker on --adapter
`build(**kv)` of an adapter returns (Source, Subject, Task); a value is an int / float / true / false only when written as one, else a string ('0061' keeps one).
RUN is a run name below <out>/quant or a run directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from typing import Any, Dict

_INT = re.compile(r"[+-]?(0|[1-9][0-9]*)")
_FLOAT = re.compile(r"[+-]?([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?|[+-]?[0-9]+[eE][+-]?[0-9]+")


def _auto(v: str) -> Any:
    """the value of --arg K=V: an int or a float only when it is written as one (an id like 0061 or 2020_01_03, nan and inf stay
    strings); true / false; quotes ('0061') keep a string whatever it looks like"""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    if _INT.fullmatch(v):
        return int(v)
    if _FLOAT.fullmatch(v):
        return float(v)
    return {"true": True, "false": False}.get(v.lower(), v)


def load_adapter(spec: str, kv: Dict[str, Any]):
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn.isidentifier():
        raise SystemExit("--adapter must be module:function or path/to/file.py:function")
    sys.path.insert(0, os.getcwd())
    if mod.endswith(".py"):                                          # a file: its folder joins the path, the file is imported by name
        if not os.path.isfile(mod):
            raise SystemExit(f"adapter file not found: {mod}")
        sys.path.append(os.path.dirname(os.path.abspath(mod)))
        mod = os.path.splitext(os.path.basename(mod))[0]
    built = getattr(importlib.import_module(mod), fn)(**kv)
    if len(built) != 3:
        raise SystemExit("an adapter's build() must return (source, subject, task)")
    return built


def _headline(suite, table: Dict[str, float]) -> Dict[str, float]:
    """the headline numbers of every metric out of a full table"""
    keep = {f"{m.name}.{k}" for m in suite.task.metrics() for k in m.headline}
    return {k: v for k, v in table.items() if k in keep}


def _hist(name: str):
    return {"warmup": None, "full": "full", "none": "none"}[name]


def _fmt_table(d: Dict[str, float]) -> str:
    w = max((len(k) for k in d), default=10)
    return "\n".join(f"{k.ljust(w)}  {v:.4f}" if isinstance(v, float) else f"{k.ljust(w)}  {v}" for k, v in sorted(d.items()))


def _qual_print(res) -> None:
    for r in (res if isinstance(res, list) else [res]):
        print(f"{r.ref}")
        for v, p in r.images.items():
            print(f"  {v}: {p}")
        if r.taps_npz:
            print(f"  taps: {r.taps_npz}")
        if r.scores:
            print("  scores: " + "  ".join(f"{k}={v:.3f}" for k, v in r.scores.items() if v == v))
        for f in r.flags:
            print(f"  FLAG {f['status']}: {f['name']} {f['note']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalsuite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", help="module:function returning (source, subject, task)")
    p.add_argument("--arg", action="append", default=[], metavar="K=V", help="keyword argument for build() (repeatable)")
    p.add_argument("--out", default="eval_out")
    p.add_argument("--state", default="carry", choices=["carry", "reset", "warmup"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").add_argument("only", nargs="*", help="run only the checks whose name contains one of these words")
    sub.add_parser("selftest-adapter")
    ls = sub.add_parser("list")
    ls.add_argument("scene", nargs="?")
    q = sub.add_parser("qual").add_subparsers(dest="mode", required=True)
    f = q.add_parser("frame")
    f.add_argument("spec")
    f.add_argument("--view", nargs="+", default=["overview"])
    f.add_argument("--taps", nargs="*", default=None)
    f.add_argument("--history", default="warmup", choices=["warmup", "full", "none"], help="frames fed before the frame")
    d = q.add_parser("debug")
    d.add_argument("spec")
    d.add_argument("--taps", nargs="*", default=None)
    d.add_argument("--history", default="warmup", choices=["warmup", "full", "none"])
    r = q.add_parser("random")
    r.add_argument("n", type=int)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--view", nargs="+", default=["overview"])
    r.add_argument("--taps", nargs="*", default=None)
    v = q.add_parser("videos")
    v.add_argument("m", type=int, nargs="?", default=1)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--view", default=None)
    v.add_argument("--fps", type=float, default=4.0)
    v.add_argument("--max-frames", type=int, default=None)
    v.add_argument("--scenes", nargs="*", default=None)
    n = sub.add_parser("quant").add_subparsers(dest="mode", required=True)
    run = n.add_parser("run")
    run.add_argument("--n-scenes", type=int, default=None)
    run.add_argument("--scenes", nargs="*", default=None)
    run.add_argument("--spread", type=int, default=None, help="N frames evenly spread over all frames (state reset)")
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--name", default=None)
    for name in ("table", "per-scene", "report", "outliers", "trace"):
        a = n.add_parser(name)
        a.add_argument("run")
        if name == "trace":
            a.add_argument("label")
        if name == "table":
            a.add_argument("--all", action="store_true", help="every number (default: the headline numbers of each metric)")
        if name == "report":
            a.add_argument("--render", type=int, default=6)
        if name == "outliers":
            a.add_argument("--k", type=int, default=10)
            a.add_argument("--z", type=float, default=3.5)
            a.add_argument("--min-support", type=float, default=1.0)
    c = n.add_parser("compare", help="paired scene bootstrap: diff = b - a = variant - control")
    c.add_argument("run_a", help="the control (reference)")
    c.add_argument("run_b", help="the variant")
    c.add_argument("--boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--force", action="store_true", help="compare even if the runs differ in source or task (dataset, split, GT version, cache)")
    t = n.add_parser("timing")
    t.add_argument("--frame", default=None)
    t.add_argument("--n", type=int, default=30)
    t.add_argument("--run", default=None, help="also store the result in this run (cost.bench), where `quant pareto` finds it")
    t.add_argument("--force", action="store_true", help="attach the benchmark to --run even if the Subject / Source are not those of the run")
    pa = n.add_parser("pareto")
    pa.add_argument("runs", nargs="+", help="runs of the same task; quality is taken on the frames all of them have")
    pa.add_argument("--y", default=None, help="quality axis, 'metric.key' (default: the first headline number)")
    pa.add_argument("--cost", nargs="+", default=["ms", "gmacs"], choices=["ms", "gmacs"], help="cost axes")
    pa.add_argument("--force", action="store_true", help="even if the runs differ in source or task")
    pr = sub.add_parser("probe").add_subparsers(dest="mode", required=True)
    for name in ("budget", "situations", "sample", "scan"):
        a = pr.add_parser(name)
        a.add_argument("run")
        if name == "sample":
            a.add_argument("situation", help="a situation name from `probe situations RUN`, e.g. 'dominated by fn.gap'")
            a.add_argument("-n", type=int, default=6)
        if name in ("sample", "scan"):
            a.add_argument("--seed", type=int, default=0)
        if name == "scan":
            a.add_argument("-k", type=int, default=3, help="random examples per situation")
            a.add_argument("--top", type=int, default=6, help="how many situations (by excess error)")
            a.add_argument("--no-render", action="store_true")
        if name == "situations":
            a.add_argument("--min-frames", type=int, default=3)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "selftest":
        from .selftest import main as selftest
        return selftest(a.only)
    if not a.adapter:
        raise SystemExit("--adapter module:function is required")
    from .suite import EvalSuite
    bad = [x for x in a.arg if "=" not in x]
    if bad:
        raise SystemExit(f"--arg expects K=V, got {bad}")
    kv = {k: _auto(v) for k, v in (x.split("=", 1) for x in a.arg)}
    source, subject, task = load_adapter(a.adapter, kv)
    suite = EvalSuite(source, subject, task, out=a.out, state=a.state)
    try:
        if a.cmd == "selftest-adapter":
            from .check import check_adapter, format_results
            res = check_adapter(suite)
            print(format_results(res))
            return 0 if all(r["ok"] for r in res) else 1
        if a.cmd == "list":
            if a.scene:
                for fr in source.frames(a.scene):
                    print(fr.label, fr.key)
            else:
                for s in source.scenes():
                    print(s, len(source.frames(s)))
        elif a.cmd == "qual":
            if a.mode == "frame":
                _qual_print(suite.qual.frame(a.spec, a.view, a.taps, history=_hist(a.history)))
            elif a.mode == "debug":
                _qual_print(suite.qual.debug(a.spec, a.taps, history=_hist(a.history)))
            elif a.mode == "random":
                _qual_print(suite.qual.random(a.n, a.seed, a.view, a.taps))
            else:
                for o in suite.qual.videos(a.m, a.seed, a.view, a.fps, a.max_frames, a.scenes):
                    print(f"{o['scene']}: {o['video']['path']} ({o['video']['codec']}, {o['frames']} frames), scores: {o['scores_strip']}")
        elif a.cmd == "probe":
            pb = suite.probe
            if a.mode == "budget":
                b = pb.budget(a.run)
                print(_fmt_table({f"{x['cause']} ({x['target']})": x["amount"] for x in b}))
                print("where to work:", {k: round(v, 3) for k, v in pb.targets(b).items()})
            elif a.mode == "situations":
                for s in pb.situations(a.run, a.min_frames):
                    print(f"{s['excess']:+.2f}  {s['name']:38s} n={s['n']:<4d} of all {s['share']:.0%}  error share {s['error_share']:.0%}  scenes {s['scenes']}  ({s['kind']}: {s['desc']})")
            elif a.mode == "sample":
                for d in pb.sample(a.run, a.situation, a.n, a.seed):
                    print(f"{d['label']}: error {d['error']:.1f}, mostly {d['top_cause'] or '-'}  {d['images'].get('overview', '')}")
            else:
                res = pb.scan(a.run, a.k, a.top, a.seed, render=not a.no_render)
                print(res["index"])
        else:
            qt = suite.quant
            if a.mode == "run":
                frames = None
                state = None
                if a.spread:
                    if a.n_scenes or a.max_frames:
                        raise SystemExit("--spread chooses the frames itself: it cannot be combined with --n-scenes or --max-frames")
                    from .sampling import Sampler
                    frames = Sampler(source, a.seed).spread(a.spread, a.scenes)
                    state = "reset"
                st = qt.run(scenes=a.scenes if not a.spread else None, n_scenes=a.n_scenes, seed=a.seed, max_frames=a.max_frames,
                            frames=frames, state=state, name=a.name)
                print(f"run {st.path}: {len(st.frames)} frames, {len(st.scene_names())} scenes")
                print(_fmt_table(_headline(suite, qt.table(st))))
                c = st.manifest.get("cost") or {}
                if c.get("run_ms"):
                    r = c["run_ms"]
                    print(f"ms per frame in this run: median {r['median_ms']:.2f}, p90 {r['p90_ms']:.2f} (n {r['n']}, load {r['load'].get('load1', '?')}; "
                          f"an idle benchmark: quant timing --run {st.manifest['name']})")
                if (c.get("ops") or {}).get("gmacs") is not None:
                    print(f"GMACs per frame: {c['ops']['gmacs']:.4g} ({c['ops'].get('method', '?')})")
            elif a.mode == "table":
                print(_fmt_table(qt.table(a.run) if a.all else _headline(suite, qt.table(a.run))))
            elif a.mode == "per-scene":
                for sc, d in qt.per_scene(a.run).items():
                    print(sc, "  ".join(f"{k}={v:.3f}" for k, v in d.items()))
            elif a.mode == "outliers":
                res = qt.outliers(a.run, a.k, a.z, a.min_support)
                print(json.dumps(res, indent=1, default=str))
            elif a.mode == "trace":
                _qual_print(qt.trace(a.run, a.label))
            elif a.mode == "report":
                print(qt.report(a.run, a.render))
            elif a.mode == "compare":
                print(json.dumps(qt.compare(a.run_a, a.run_b, a.boot, a.seed, force=a.force), indent=1))
            elif a.mode == "pareto":
                from .cost import format_table
                res = qt.pareto(a.runs, a.y, a.cost, force=a.force)
                print(format_table(res))
                print(f"picture: {res['png']}" if res["png"] else "no cost to draw: no run has ms or GMACs")
            elif a.mode == "timing":
                print(json.dumps(qt.timing(a.frame, a.n, a.run, a.force), indent=1))
    finally:
        suite.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

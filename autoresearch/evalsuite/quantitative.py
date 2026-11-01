"""The quantitative suite: numbers over all videos together, and the way back from a number to the frames it comes from.

    run(...)                 every chosen scene through the Subject in time order -> a RunStore (counts of every Metric per frame,
                             per-frame scores, flags, ms); resumable
    table(run)               the Metrics' reduce() over all frames: the dataset-level numbers
    per_scene(run)           the same per scene
    outliers(run)            where the numbers come from: (1) the scenes that drag a headline number most (exact leave-one-out),
                             (2) per score the frames far outside the others (robust z, only frames with enough support),
                             each listed frame with its own drag on the headline numbers, (3) plausibility flags;
                             every row carries the frame label and the command that draws it
    trace(run, label)        draw such a frame (overview + debug) through the qualitative suite
    compare(a, b)            paired scene bootstrap of the headline numbers between two stored runs (a = control, b = variant: b - a;
                             same frames, same source and task), no network
    report(run)              html: tables, outliers with pictures of the worst frames
    timing(run=)             ms per frame of the Subject (only valid on an idle machine; the load is recorded); run() itself records
                             the ms per frame of its own frames and Subject.ops (cost.py), pareto() sets runs against each other
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .contracts import FrameRef, Metric
from .render import report as html
from .runner import Runner
from .sampling import Sampler
from .stats import bootstrap_scenes, loo_influence, robust_z
from .store import RunStore, _identity_differences, warn_gained


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


class Quantitative:
    def __init__(self, suite):
        self.s = suite
        self.dir = os.path.join(suite.out, "quant")

    # -------------------------------------------------------------------------------------- run
    def run_dir(self, run: str) -> str:
        return run if os.path.isfile(os.path.join(run, "manifest.json")) else os.path.join(self.dir, run)

    def open(self, run: Union[str, RunStore]) -> RunStore:
        return run if isinstance(run, RunStore) else RunStore.open(self.run_dir(run))

    def _metrics(self) -> Dict[str, Metric]:
        listed = self.s.task.metrics()
        ms = {m.name: m for m in listed}
        if len(ms) != len(listed):
            raise ValueError(f"the metric names of the task are not unique: {sorted(m.name for m in listed)}")
        return ms

    def run(self, scenes: Optional[Sequence[str]] = None, n_scenes: Optional[int] = None, seed: int = 0,
            max_frames: Optional[int] = None, frames: Optional[Sequence[FrameRef]] = None, state: Optional[str] = None,
            name: Optional[str] = None, resume: bool = True, progress: int = 25, on_frame=None) -> RunStore:
        """Run the chosen scenes (default: all; n_scenes: that many random scenes; max_frames: the first frames of each scene)
        or the given frames (grouped by scene, in time order) through the Subject and store the counts of every Metric.
        state 'carry' (default of the suite) | 'reset' (every frame independent) | 'warmup'. on_frame(sample, prediction): optional callback
        for EVERY frame run (also frames a resumed run already has stored: the Subject runs them anyway). -> the closed RunStore."""
        src, subj, task = self.s.source, self.s.subject, self.s.task
        state = state or self.s.state
        if frames is not None:
            by: Dict[str, List[FrameRef]] = {}
            for f in sorted(frames):
                by.setdefault(f.scene, []).append(f)
            plan = by
            if state == "carry":
                for sc, fr in plan.items():
                    if [f.index for f in fr] != list(range(fr[0].index, fr[0].index + len(fr))):
                        raise ValueError(f"{sc}: state='carry' needs consecutive frames; use state='reset' for a spread subset")
        else:
            chosen = list(scenes) if scenes else (Sampler(src, seed).scenes(n_scenes) if n_scenes else src.scenes())
            plan = {sc: src.frames(sc)[: max_frames or None] for sc in chosen}
        if not any(plan.values()):
            raise ValueError("no frames to run (an empty frame list, unknown scenes or max_frames that leaves nothing)")
        metrics = self._metrics()
        desc = subj.describe()
        run_name = name or _slug(f"{desc.get('name', type(subj).__name__)}_{state}_{time.strftime('%Y%m%d_%H%M%S')}")
        manifest = {"name": run_name, "state": state, "source": src.describe(), "subject": desc, "task": task.name,
                    "metrics": {m.name: {"headline": list(m.headline), "better": {k: m.direction(k) for k in m.headline}} for m in metrics.values()},
                    "selection": {"scenes": list(plan), "n_frames_planned": sum(len(v) for v in plan.values()),
                                  "max_frames": max_frames if frames is None else None, "seed": seed,     # frames= chooses the frames: no limit applied
                                  "frames_sha": hashlib.sha1("\n".join(f.label for v in plan.values() for f in v).encode()).hexdigest()[:12]}}
        store = RunStore.create(os.path.join(self.dir, run_name), manifest, resume=resume)
        done = store.done()
        n_before = len(store.frames)
        cost = store.manifest.setdefault("cost", {})
        if "ops" not in cost:
            ops = self._ops(next((r for refs in plan.values() for r in refs), None), run_name)
            if ops:
                cost["ops"] = ops
        load0 = machine_load()                                # before the first frame: what else was running while we started
        runner = Runner(src, subj, state)
        t0, n = time.perf_counter(), 0
        total = manifest["selection"]["n_frames_planned"]
        finished = False
        try:
            for sc, refs in plan.items():
                if all(r.label in done for r in refs):
                    continue
                for sample, pred in runner.iter_scene(sc, refs):
                    n += 1
                    if on_frame is not None:
                        on_frame(sample, pred)
                    if sample.ref.label in done:
                        continue
                    counts, scores, support = {}, {}, {}
                    for m in metrics.values():
                        c = m.frame(sample, pred)
                        counts[m.name] = c
                        scores.update({f"{m.name}.{k}": float(v) for k, v in m.score(c).items()})
                        support.update({f"{m.name}.{k}": float(v) for k, v in m.support(c).items()})
                    flags = [dict(name=f.name, status=f.status, value=_plain(f.value), note=f.note)
                             for f in task.checks(sample, pred) if f.status != "ok"]
                    info = {k: v for k, v in pred.info.items() if isinstance(v, (int, float, str, bool))}
                    pr = task.probe(sample, pred)
                    if pr:                                                       # descriptors and error causes for the probing
                        info["probe"] = {k: round(float(v), 4) for k, v in pr.items()}
                    store.add(sample.ref, counts, scores, support, flags, info)
                    if progress and n % progress == 0:
                        dt = time.perf_counter() - t0
                        print(f"[quant {run_name}] {n}/{total} frames, {dt / n:.2f} s per frame, eta {dt / n * (total - n):.0f} s",
                              flush=True)
            finished = True
        finally:                                              # an interrupted run keeps a consistent prefix and can be resumed
            store.flush()                                     # the frames of this session are safe before anything slow (nvidia-smi) happens
            ms = [float(fr["info"]["predict_ms"]) for fr in store.frames if "predict_ms" in fr.get("info", {})]
            if ms and len(store.frames) > n_before:           # every stored frame, also those of earlier sessions of a resumed run
                a = np.sort(np.asarray(ms))
                cost["run_ms"] = {"median_ms": float(np.median(a)), "p90_ms": float(a[int(0.9 * (len(a) - 1))]),
                                  "mean_ms": float(a.mean()), "n": len(a), "load": load0, "load_end": machine_load()}
            store.manifest["wall_s"] = round(store.manifest.get("wall_s", 0) + time.perf_counter() - t0, 1)
            store.close(finished=finished)
        return store

    def _ops(self, ref, run_name: str) -> Dict[str, Any]:
        """Subject.ops on the first frame of the run; an estimate that fails must not stop a run (it is tried again on a resume)"""
        if ref is None:
            return {}
        try:
            ops = self.s.subject.ops(self.s.source.load(ref)) or {}
            if not isinstance(ops, dict):
                raise TypeError(f"ops() must return a dict with 'gmacs', not {type(ops).__name__}")
            ops = json.loads(json.dumps(ops, default=_json_number))              # what the manifest holds on disk (numpy scalars become numbers)
            if ops and not isinstance(ops.get("gmacs"), (int, float)):
                raise TypeError("ops() must return a number under 'gmacs'")
            return ops
        except Exception as e:                                                         # noqa: BLE001
            print(f"[quant {run_name}] operations not counted: {type(e).__name__}: {e}", flush=True)
            return {}
        finally:
            self.s.subject.reset()

    # -------------------------------------------------------------------------------------- numbers
    def table(self, run: Union[str, RunStore], labels: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """dataset-level numbers: 'metric.key' -> value over all frames of the run (or `labels`)"""
        st = self.open(run)
        out: Dict[str, float] = {}
        for name, m in self._metrics().items():
            out.update({f"{name}.{k}": float(v) for k, v in m.reduce(st.counts(name, labels)).items()})
        return out

    def per_scene(self, run: Union[str, RunStore]) -> Dict[str, Dict[str, float]]:
        """scene -> {'metric.key': value} for the headline keys"""
        st = self.open(run)
        res: Dict[str, Dict[str, float]] = {}
        for name, m in self._metrics().items():
            keys = list(m.headline) or None
            for sc, cs in st.by_scene(name).items():
                res.setdefault(sc, {}).update({f"{name}.{k}": float(v) for k, v in m.reduce(cs, keys=keys).items()
                                              if keys is None or k in keys})
        return res

    # -------------------------------------------------------------------------------------- outliers
    def outliers(self, run: Union[str, RunStore], k: int = 10, z: float = 3.5, min_support: float = 1.0) -> Dict[str, List[dict]]:
        """Where the numbers come from. Returns {'scenes': [...], 'frames': [...], 'flags': [...]}.
        scenes: per headline number the k scenes whose removal improves it most ('drag' > 0 = the scene pulls the number the
          wrong way), exact leave-one-out over the stored counts.
        frames: per score (a Metric's per-frame scalar) the k frames furthest outside the others in the bad direction, by
          robust z (median / MAD) among frames with support >= min_support; |z| >= z is marked 'outlier'; each row also gives
          the frame's own drag on the headline numbers of its metric.
        flags: frames carrying plausibility flags, worst first. Every row has 'label' (or 'scene') and 'view' = the command."""
        st = self.open(run)
        metrics = self._metrics()
        res: Dict[str, List[dict]] = {"scenes": [], "frames": [], "flags": []}
        # (1) scenes
        for name, m in metrics.items():
            hk = list(m.headline)
            if not hk:
                continue
            by_scene = st.by_scene(name)
            infl = loo_influence(by_scene, lambda c, _m=m, _k=hk: _m.reduce(c, keys=_k), hk)
            for key in hk:
                sign = 1.0 if m.direction(key) == "high" else -1.0
                rows = [{"kind": "scene", "metric": f"{name}.{key}", "scene": sc, "drag": -sign * v[key], "frames": len(by_scene[sc]),
                         "view": f"qual videos 1 --scenes {sc}"} for sc, v in infl.items()
                        if v[key] == v[key]]
                rows.sort(key=lambda r: -r["drag"])
                res["scenes"] += rows[:k]
        # (2) frames
        frames = st.frames
        for name, m in metrics.items():
            score_keys = sorted({key for fr in frames for key in fr["scores"] if key.startswith(name + ".")})
            hk = list(m.headline)
            counts = st.counts(name)
            full = m.reduce(counts, keys=hk) if hk else {}                    # the run's own numbers, the same for every frame
            for sk in score_keys:
                vals = np.array([fr["scores"].get(sk, np.nan) for fr in frames], float)
                sup = np.array([fr["support"].get(sk, 1.0) for fr in frames], float)
                ok = np.isfinite(vals) & (sup >= min_support)
                if ok.sum() < 4:
                    continue
                zz = np.full(len(frames), np.nan)
                zz[ok] = robust_z(vals[ok])
                sign = 1.0 if m.direction(sk.split(".", 1)[1]) == "high" else -1.0
                order = [i for i in np.argsort(sign * zz) if ok[i]][:k]            # most negative (bad) z first for 'high'
                for i in order:
                    row = {"kind": "frame", "metric": sk, "label": frames[i]["label"], "scene": frames[i]["scene"],
                           "value": float(vals[i]), "z": float(zz[i]), "support": float(sup[i]),
                           "outlier": bool(-sign * zz[i] >= z), "view": f"qual frame {frames[i]['label']} --view overview debug"}
                    if hk:
                        rest = m.reduce([c for j, c in enumerate(counts) if j != i], keys=hk)
                        row["drag"] = {key: float(-(1.0 if m.direction(key) == "high" else -1.0) * (full[key] - rest[key]))
                                       for key in hk if full[key] == full[key] and rest[key] == rest[key]}
                    res["frames"].append(row)
        # (3) flags
        rank = {"fail": 0, "warn": 1}
        for fr in frames:
            for f in fr["flags"]:
                res["flags"].append({"kind": "flag", "label": fr["label"], "scene": fr["scene"], "status": f["status"], "name": f["name"],
                                     "note": f["note"], "view": f"qual frame {fr['label']} --view overview debug"})
        res["flags"].sort(key=lambda r: (rank.get(r["status"], 2), r["label"]))
        res["flags"] = res["flags"][: max(k * 3, 30)]
        return res

    def trace(self, run: Union[str, RunStore], label: str, views: Sequence[str] = ("overview", "debug")):
        """draw one frame of a stored run (same Source / Subject as the suite was built with)"""
        st = self.open(run)
        fr = st.frame(label)
        state = st.manifest.get("state", "carry")
        history = {"carry": "full", "warmup": None, "reset": "none"}[state]                          # the exact replay of the run
        return self.s.qual.frame((fr["scene"], fr["index"]), views, out_dir=os.path.join(st.path, "trace"), history=history, state=state)

    # -------------------------------------------------------------------------------------- compare
    def compare(self, run_a: Union[str, RunStore], run_b: Union[str, RunStore], boot: int = 1000, seed: int = 0, force: bool = False,
                ignore: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
        """Paired scene bootstrap of the headline numbers on the frames both runs have: run_a = the CONTROL (reference), run_b = the
        VARIANT, diff = b - a = variant - control (the sign autoresearch/tools/decide.py reads, the same as the lane project's
        SB.paired(variant, control)). -> {'metric.key': {a, b, diff, lo, hi, p_b_better (direction-aware), p_diff_gt0, n_valid, ...}}.
        Both runs must come from the same Source and Task (their manifests' `source` and `task` entries: dataset, split, ground-truth
        version, caches); otherwise the counts are of different questions and the comparison is refused (force=True compares anyway).
        ignore: description keys that are the arms' difference, not the question ({"source": ("feature_cache",)}: a run fed with
        other features answers the same question on the same frames and ground truth)."""
        a, b = self.open(run_a), self.open(run_b)
        same_question(a, b, force=force, ignore=ignore)
        common = set(a.labels()) & set(b.labels())
        if not common:
            raise ValueError("the runs have no frame in common")
        out: Dict[str, Any] = {"n_common_frames": len(common), "only_a": len(set(a.labels()) - common),
                               "only_b": len(set(b.labels()) - common), "metrics": {}}
        for name, m in self._metrics().items():
            hk = list(m.headline)
            if not hk:
                continue
            ca, cb = _by_scene(a, name, common), _by_scene(b, name, common)
            for key, v in bootstrap_scenes(ca, cb, m.reduce, hk, n=boot, seed=seed, better={k: m.direction(k) for k in hk}).items():
                v["better"] = m.direction(key)
                out["metrics"][f"{name}.{key}"] = v
        return out

    # -------------------------------------------------------------------------------------- report
    def report(self, run: Union[str, RunStore], render: int = 6, k: int = 8) -> str:
        """<run>/report.html: manifest, headline numbers, scenes sorted by drag, outliers, the pictures of the worst frames"""
        st = self.open(run)
        tab = self.table(st)
        osr = self.outliers(st, k=k)
        man = st.manifest
        sections = [("run", html.table([["field", "value"], ["name", man.get("name")], ["state", man.get("state")],
                                         ["frames", len(st.frames)], ["scenes", len(st.scene_names())],
                                         ["subject", json.dumps(man.get("subject"), default=str)[:200]],
                                         ["source", json.dumps(man.get("source"), default=str)[:200]],
                                         ["commit", man.get("commit")], ["wall s", man.get("wall_s")]] + _cost_rows(man), left_cols=(0, 1)))]
        head = []
        for name, m in self._metrics().items():
            head += [[f"{name}.{key}", tab.get(f"{name}.{key}", float("nan")), m.direction(key)] for key in m.headline]
        sections.append(("headline numbers (all frames of the run)", html.table([["number", "value", "better"]] + head)))
        sections.append(("all numbers", html.table([["number", "value"]] + [[kk, vv] for kk, vv in sorted(tab.items())])))
        sc_rows = [["metric", "scene", "drag", "frames", "how to look"]] + [[r["metric"], r["scene"], r["drag"], r["frames"], r["view"]]
                                                                              for r in osr["scenes"]]
        sections.append(("scenes that drag the headline numbers most (leave-one-out; positive = hurts)",
                         html.table(sc_rows, left_cols=(0, 1, 4))))
        fr_rows = [["metric", "frame", "value", "z", "support", "outlier", "how to look"]]
        fr_rows += [[r["metric"], r["label"], r["value"], r["z"], r["support"], r["outlier"], r["view"]] for r in osr["frames"]]
        sections.append(("frames furthest outside the others (per score, bad direction)", html.table(fr_rows, left_cols=(0, 1, 6))))
        if osr["flags"]:
            sections.append(("plausibility flags", html.table([["status", "frame", "flag", "note"]] +
                                                                [[r["status"], r["label"], r["name"], r["note"]] for r in osr["flags"]],
                                                                left_cols=(0, 1, 2, 3), status_col=0)))
        shown, thumbs = set(), []
        for r in osr["frames"]:
            if len(shown) >= render:
                break
            if r["label"] in shown:
                continue
            shown.add(r["label"])
            q = self.trace(st, r["label"], ("overview",))
            thumbs.append(html.image(q.images["overview"], st.path, 760, f"{r['label']}: {r['metric']} = {r['value']:.3f} (z {r['z']:.1f})"))
        if thumbs:
            sections.append(("the worst frames (pictures)", "".join(thumbs)))
        path = html.write(os.path.join(st.path, "report.html"), f"quantitative report: {man.get('name')}", sections)
        self.export_csv(st)
        return path

    def export_csv(self, run: Union[str, RunStore]) -> str:
        """frames.csv: one row per frame with all its scores (for any external tool)"""
        st = self.open(run)
        keys = sorted({k for fr in st.frames for k in fr["scores"]})
        path = os.path.join(st.path, "frames.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "scene", "index", "key", "flags"] + keys)
            for fr in st.frames:
                w.writerow([fr["label"], fr["scene"], fr["index"], fr["key"], len(fr["flags"])] + [fr["scores"].get(k, "") for k in keys])
        return path

    # -------------------------------------------------------------------------------------- timing
    def pareto(self, runs: Sequence[Union[str, RunStore]], y: Optional[str] = None, costs: Sequence[str] = ("ms", "gmacs"),
               out: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """quality (`y` = 'metric.key', default the first headline number) against ms and GMACs per frame for several runs, on the
        frames all of them have; marks the runs no other run beats on both (see cost.py); writes quant/pareto_<y>.png and .json;
        the runs must share source and task (force=True skips that check)"""
        from . import cost
        return cost.pareto(self, runs, y, costs, out, force=force)

    def timing(self, spec: Any = None, n: int = 30, run: Optional[Union[str, RunStore]] = None, force: bool = False) -> Dict[str, Any]:
        """ms per frame of the Subject on one frame (the first frame of the first scene, or `spec`). Only valid on an idle
        machine: the load average (and the GPU utilisation, if nvidia-smi exists) are read BEFORE the benchmark and recorded next to
        the numbers. With `run` the result is also stored in that run's manifest (cost.bench), where `pareto` finds it; the Subject,
        Source and Task of this suite must be those of the run (force=True skips the check)."""
        st = self.open(run) if run is not None else None
        if st is not None and not force:
            gained: List[str] = []
            diff = _identity_differences(st.manifest, {"source": self.s.source.describe(), "subject": self.s.subject.describe(),
                                                       "task": self.s.task.name}, ("source", "subject", "task"), gained=gained)
            if not diff:
                warn_gained(gained, f"timing attached to run {st.manifest.get('name')}")
            if diff:
                raise ValueError(f"this suite is not the one that made run {st.manifest.get('name')} ({'; '.join(diff)}): a benchmark of "
                                 "another network must not be attached to it (force=True does it anyway)")
        ref = self.s.source.find(spec) if spec is not None else self.s.source.frames(self.s.source.scenes()[0])[0]
        sample, _ = Runner(self.s.source, self.s.subject, self.s.state).run_to(ref)
        load = machine_load()
        res = self.s.subject.benchmark(sample, n=n)
        res["load"] = load
        res["frame"] = ref.label
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "timing.json"), "w") as f:
            json.dump(res, f, indent=1)
        if st is not None:
            st.set_cost("bench", dict(res))
        return res


def same_question(*stores: RunStore, force: bool = False, ignore: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """refuse runs whose manifests differ in `source` or `task` (dataset, split, ground-truth version, caches: the bev_fusion GT signature
    trap) unless force=True; the first run is the reference. ignore: {entry: keys} that may differ (the inputs the arms vary)"""
    if force or len(stores) < 2:
        return
    bad = []
    for st in stores[1:]:
        gained: List[str] = []
        diff = _identity_differences(stores[0].manifest, st.manifest, ("source", "task"), gained=gained, either=True, ignore=ignore)
        warn_gained(gained, f"comparing {st.manifest.get('name', st.path)} with {stores[0].manifest.get('name', stores[0].path)}")
        if diff:
            bad.append(f"{st.manifest.get('name', st.path)} vs {stores[0].manifest.get('name', stores[0].path)}: {'; '.join(diff)}")
    if bad:
        raise ValueError("the runs do not come from the same source and task, so their numbers answer different questions ("
                         + " | ".join(bad) + "); evaluate both on the same data, or force=True (--force) to compare anyway")


def _cost_rows(man: Dict[str, Any]) -> List[list]:
    """the cost entries of a manifest as rows of the report's run table"""
    c, rows = man.get("cost") or {}, []
    for key, label in (("run_ms", "ms per frame, measured in the run"), ("bench", "ms per frame, benchmark")):
        if c.get(key):
            rows.append([label, f"median {c[key]['median_ms']:.2f}, p90 {c[key]['p90_ms']:.2f} (n {c[key]['n']}, load {c[key].get('load', {}).get('load1', '?')})"])
    if (c.get("ops") or {}).get("gmacs") is not None:
        rows.append(["GMACs per frame", f"{c['ops']['gmacs']:.4g} ({c['ops'].get('method', '?')})"])
    return rows


def _by_scene(st: RunStore, metric: str, labels: set) -> Dict[str, list]:
    res: Dict[str, list] = {}
    for fr, c in zip(st.frames, st.counts(metric)):
        if fr["label"] in labels:
            res.setdefault(fr["scene"], []).append(c)
    return res


def _json_number(o: Any) -> Any:
    """json.dumps default: numpy scalars and arrays become plain numbers and lists (the manifest's own default=str would make strings of them)"""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _plain(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


def machine_load() -> Dict[str, Any]:
    """the state of the machine now: load average, CPU count, host name, GPU utilisation / memory / name (what a timing must say about itself)"""
    import platform
    out: Dict[str, Any] = {"host": platform.node()}
    try:
        out["load1"] = os.getloadavg()[0]
        out["cpus"] = os.cpu_count()
    except (OSError, AttributeError):
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,name", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        out["gpu_util_percent"], out["gpu_mem_mib"], out["gpu_name"] = float(r[0]), float(r[1]), r[2].strip()
    except Exception:                                                              # noqa: BLE001
        pass
    return out

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
    compare(a, b)            paired scene bootstrap of the headline numbers between two stored runs (same frames), no network
    report(run)              html: tables, outliers with pictures of the worst frames
    timing()                 ms per frame of the Subject (only valid on an idle machine; the load is recorded)
"""
from __future__ import annotations

import csv
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
from .store import RunStore


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
        ms = {m.name: m for m in self.s.task.metrics()}
        return ms

    def run(self, scenes: Optional[Sequence[str]] = None, n_scenes: Optional[int] = None, seed: int = 0,
            max_frames: Optional[int] = None, frames: Optional[Sequence[FrameRef]] = None, state: Optional[str] = None,
            name: Optional[str] = None, resume: bool = True, progress: int = 25) -> RunStore:
        """Run the chosen scenes (default: all; n_scenes: that many random scenes; max_frames: the first frames of each scene)
        or the given frames (grouped by scene, in time order) through the Subject and store the counts of every Metric.
        state 'carry' (default of the suite) | 'reset' (every frame independent). -> the closed RunStore."""
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
        metrics = self._metrics()
        desc = subj.describe()
        run_name = name or _slug(f"{desc.get('name', type(subj).__name__)}_{state}_{time.strftime('%Y%m%d_%H%M%S')}")
        manifest = {"name": run_name, "state": state, "source": src.describe(), "subject": desc, "task": task.name,
                    "metrics": {m.name: {"headline": list(m.headline), "better": dict(m.better)} for m in metrics.values()},
                    "selection": {"scenes": list(plan), "n_frames_planned": sum(len(v) for v in plan.values()),
                                  "max_frames": max_frames, "seed": seed}}
        store = RunStore.create(os.path.join(self.dir, run_name), manifest, resume=resume)
        done = store.done()
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
                    store.add(sample.ref, counts, scores, support, flags, info)
                    if progress and n % progress == 0:
                        dt = time.perf_counter() - t0
                        print(f"[quant {run_name}] {n}/{total} frames, {dt / n:.2f} s per frame, eta {dt / n * (total - n):.0f} s",
                              flush=True)
            finished = True
        finally:                                              # an interrupted run keeps a consistent prefix and can be resumed
            store.manifest["wall_s"] = round(store.manifest.get("wall_s", 0) + time.perf_counter() - t0, 1)
            store.close(finished=finished)
        return store

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
                         "view": f"quant outliers --scene {sc}  |  qual videos --scenes {sc}"} for sc, v in infl.items()
                        if v[key] == v[key]]
                rows.sort(key=lambda r: -r["drag"])
                res["scenes"] += rows[:k]
        # (2) frames
        frames = st.frames
        for name, m in metrics.items():
            score_keys = sorted({key for fr in frames for key in fr["scores"] if key.startswith(name + ".")})
            hk = list(m.headline)
            counts = st.counts(name)
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
                        full = m.reduce(counts, keys=hk)
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
        return self.s.qual.frame((fr["scene"], fr["index"]), views, out_dir=os.path.join(st.path, "trace"),
                                 history="full" if state == "carry" else "none", state=state)        # the exact replay of the run

    # -------------------------------------------------------------------------------------- compare
    def compare(self, run_a: Union[str, RunStore], run_b: Union[str, RunStore], boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
        """Paired scene bootstrap, b - a, of the headline numbers on the frames both runs have. -> {'metric.key': {...}}"""
        a, b = self.open(run_a), self.open(run_b)
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
            for key, v in bootstrap_scenes(ca, cb, m.reduce, hk, n=boot, seed=seed).items():
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
                                         ["commit", man.get("commit")], ["wall s", man.get("wall_s")]], left_cols=(0, 1)))]
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
    def timing(self, spec: Any = None, n: int = 30) -> Dict[str, Any]:
        """ms per frame of the Subject on one frame (the first frame of the first scene, or `spec`). Only valid on an idle
        machine: the load average (and the GPU utilisation, if nvidia-smi exists) are recorded next to the numbers."""
        ref = self.s.source.find(spec) if spec is not None else self.s.source.frames(self.s.source.scenes()[0])[0]
        sample, _ = Runner(self.s.source, self.s.subject, self.s.state).run_to(ref)
        res = self.s.subject.benchmark(sample, n=n)
        res["load"] = machine_load()
        res["frame"] = ref.label
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "timing.json"), "w") as f:
            json.dump(res, f, indent=1)
        return res


def _by_scene(st: RunStore, metric: str, labels: set) -> Dict[str, list]:
    res: Dict[str, list] = {}
    for fr, c in zip(st.frames, st.counts(metric)):
        if fr["label"] in labels:
            res.setdefault(fr["scene"], []).append(c)
    return res


def _plain(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


def machine_load() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["load1"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        out["gpu_util_percent"], out["gpu_mem_mib"] = float(r[0]), float(r[1])
    except Exception:                                                              # noqa: BLE001
        pass
    return out

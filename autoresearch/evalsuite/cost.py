"""Cost next to quality, and the Pareto front: a good system is judged by its position on the curve of quality against runtime
(and operations), not by quality alone.

A stored run carries `manifest["cost"]`:
    run_ms   ms per frame of Subject.predict() as the suite called it during `quant run`: median, p90, mean over every stored frame,
             the machine load when the run started ("load") and when it ended ("load_end"). It times whatever predict() does
             (decoding, copies) with the read-ahead thread running beside it, on a machine that may be busy: a by-product, not a benchmark
    bench    ms per frame of `quant timing --run RUN`: the benchmark on one frame (warm-up, median of n calls), the machine load read
             before it started, host and GPU name; only attached to the run that the Subject / Source / Task of the suite made
    ops      Subject.ops(sample): {"gmacs", "method": "measured" | "estimated" | "mixed", "note", "parts"}; optional

`pareto(quant, runs)` (command `quant pareto A B C`) scores the runs on the frames ALL of them have, puts them on the plane quality x
cost and marks the runs that no other run beats on both (not slower or bigger AND not worse, and better in at least one). Both
"beaten" and "not beaten" are point estimates without a noise band: a run 0.001 worse and 0.01 ms slower counts as beaten. Whether a
quality difference between neighbours is real is a question for `quant compare` and the noise band of the task.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def with_estimate(ops: Optional[Dict[str, Any]], name: str, gmacs: float, how: str) -> Dict[str, Any]:
    """Add the operations of a part that no counter sees (a custom CUDA op, numpy / numba code) to a Subject.ops result as an
    ESTIMATE with its formula (`how`), so a reader can check it: the method becomes 'mixed' (or stays 'estimated')."""
    out = dict(ops) if ops else {"gmacs": 0.0, "method": "estimated", "note": "", "parts": {}}
    parts = dict(out.get("parts", {}))
    parts[name] = float(parts.get(name, 0.0)) + float(gmacs)         # the same part added twice (a loop over levels): the parts keep adding up to the total
    method = out.get("method", "estimated")
    note = f"{out.get('note', '')}; {name}: estimated, {how}".lstrip("; ")
    out.update(gmacs=float(out.get("gmacs", 0.0)) + float(gmacs), parts=parts, method="mixed" if method == "measured" else method, note=note)
    return out


def front(cost: Sequence[float], quality: Sequence[float], high: bool = True) -> List[bool]:
    """which points are not beaten: a point is dominated when another has cost <= and quality >= (<= when lower is better), with at
    least one strict. Equal points do not dominate each other; a point without a finite cost or quality is not on the front."""
    n = len(cost)
    ok = [bool(np.isfinite(cost[i]) and np.isfinite(quality[i])) for i in range(n)]
    on = list(ok)
    for i in range(n):
        for j in range(n):
            if i == j or not ok[i] or not ok[j]:
                continue
            q_ge = quality[j] >= quality[i] if high else quality[j] <= quality[i]
            q_gt = quality[j] > quality[i] if high else quality[j] < quality[i]
            if cost[j] <= cost[i] and q_ge and (cost[j] < cost[i] or q_gt):
                on[i] = False
                break
    return on


def _ms_source(costs: List[Dict[str, Any]]) -> str:
    if all(c.get("bench") for c in costs):
        return "bench"
    if all(c.get("run_ms") for c in costs):
        return "run_ms"
    if not any(c.get("bench") or c.get("run_ms") for c in costs):
        return "none"
    return "mixed"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def pareto(quant, runs: Sequence[Any], y: Optional[str] = None, costs: Sequence[str] = ("ms", "gmacs"),
           out: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """quality `y` ('metric.key', default the first headline number) of every run on the common frames, against ms per frame and
    GMACs; -> {'rows', 'y', 'better', 'n_common_frames', 'ms_source', 'notes', 'png', 'json'} and the picture below `quant/`.
    The runs must come from the same source and task (quantitative.same_question; force=True skips the check)."""
    from .quantitative import same_question
    stores = [quant.open(r) for r in runs]
    same_question(*stores, force=force)
    common = set.intersection(*[set(s.labels()) for s in stores])
    if not common:
        raise ValueError("the runs have no frame in common")
    metrics = quant._metrics()
    if y is None:
        y = next(f"{m.name}.{m.headline[0]}" for m in metrics.values() if m.headline)
    headline = [f"{m.name}.{k}" for m in metrics.values() for k in m.headline]
    if "." not in y or y.split(".", 1)[0] not in metrics:
        raise KeyError(f"{y!r}: the quality axis is 'metric.key'; headline numbers: {headline}")
    mname, key = y.split(".", 1)
    metric, better = metrics[mname], metrics[mname].direction(key)
    available = metric.reduce(stores[0].counts(mname, common), keys=[key])
    if key not in available:
        raise KeyError(f"{y!r}: metric {mname!r} has no number {key!r}; its numbers: {sorted(available)}")
    cost = [dict(s.manifest.get("cost") or {}) for s in stores]
    src = _ms_source(cost)
    notes: List[str] = []
    if len(common) != min(len(s.labels()) for s in stores) or len({len(s.labels()) for s in stores}) > 1:
        notes.append(f"the runs have different frames: quality is computed on the {len(common)} frames all of them have")
    if "ms" in costs:
        if src == "none":
            notes.append("no run has a runtime: `quant timing --run RUN` stores one (the runs of an older version of the suite have none)")
        elif src == "mixed":
            notes.append("ms comes from different sources (the benchmark for some runs, the run itself or nothing for others): not comparable; "
                         "run `quant timing --run RUN` for every run")
        elif src == "run_ms":
            notes.append("ms is the time of predict() measured during the quantitative run (with whatever else predict does, and the read-ahead "
                         "running beside it), not an idle benchmark: `quant timing --run RUN` makes one")
        else:
            notes.append("ms are the benchmarks of `quant timing --run RUN`: valid only if the machine was idle and the same for every run")
    rows: List[Dict[str, Any]] = []
    for s, c in zip(stores, cost):
        ms_rec = (c.get("bench") if src == "bench" else (c.get("run_ms") or c.get("bench"))) or {}
        ops = c.get("ops") or {}
        load = ms_rec.get("load") or {}
        rows.append({"run": s.manifest.get("name", os.path.basename(s.path)),
                     "quality": float(metric.reduce(s.counts(mname, common), keys=[key]).get(key, float("nan"))),
                     "ms": float(ms_rec["median_ms"]) if ms_rec else None, "load1": load.get("load1"), "cpus": load.get("cpus"),
                     "host": load.get("host"), "gpu": load.get("gpu_name"),
                     "gmacs": float(ops["gmacs"]) if ops.get("gmacs") is not None else None, "ops_method": ops.get("method")})
    nan_runs = [r["run"] for r in rows if r["quality"] != r["quality"]]
    if nan_runs:
        notes.append(f"quality {y} is nan for {nan_runs} on the common frames (nothing to find there?): they are on no front")
    if "ms" in costs:
        busy = [r["run"] for r in rows if r["ms"] is not None and r["load1"] is not None and r["cpus"] and r["load1"] > 0.5 * r["cpus"]]
        if busy:
            notes.append(f"the machine was busy when {busy} were timed (load average above half the CPUs when the timing started; contention "
                         "that starts later is not seen): their ms are not trustworthy, repeat `quant timing --run RUN` on an idle machine")
        where = {(r["host"], r["gpu"]) for r in rows if r["ms"] is not None and (r["host"] or r["gpu"])}
        if len(where) > 1:
            notes.append(f"the ms were measured on different machines or GPUs {sorted(map(str, where))}: not comparable")
    axes = [a for a in costs if any(r[a] is not None for r in rows)]
    for a in axes:
        idx = [i for i, r in enumerate(rows) if r[a] is not None]
        on = front([rows[i][a] for i in idx], [rows[i]["quality"] for i in idx], better == "high")
        for r in rows:
            r["front_" + a] = False
        for i, f in zip(idx, on):
            rows[i]["front_" + a] = f
        if len(idx) < len(rows):
            notes.append(f"{len(rows) - len(idx)} run(s) have no {a}: they are not on its front")
    path = out or os.path.join(quant.dir, f"pareto_{_slug(y)}.png")
    if not path.lower().endswith(".png"):
        path += ".png"
    res = {"y": y, "better": better, "n_common_frames": len(common), "ms_source": src, "rows": rows, "notes": notes,
           "png": _plot(rows, y, better == "high", axes, path, len(common)) if axes else None}
    res["json"] = os.path.splitext(path)[0] + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(res["json"], "w") as f:
        json.dump(res, f, indent=1)
    return res


def _plot(rows, y: str, high: bool, axes: Sequence[str], path: str, n_frames: int) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"ms": "ms per frame (lower is better)", "gmacs": "GMACs per frame (lower is better)"}
    fig, axs = plt.subplots(1, len(axes), figsize=(5.4 * len(axes), 4.4), squeeze=False)
    for ax, a in zip(axs[0], axes):
        pts = [r for r in rows if r[a] is not None and np.isfinite(r["quality"])]
        fr = sorted((r for r in pts if r["front_" + a]), key=lambda r: r[a])
        ax.plot([r[a] for r in fr], [r["quality"] for r in fr], "--", color="0.55", lw=1, zorder=1)
        for r in pts:
            ax.scatter(r[a], r["quality"], s=60 if r["front_" + a] else 28, color="tab:blue" if r["front_" + a] else "0.6", zorder=2)
            ax.annotate(r["run"], (r[a], r["quality"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xlabel(labels[a])
        ax.set_ylabel(f"{y} ({'higher' if high else 'lower'} is better)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"quality against cost on {n_frames} common frames; blue = not beaten on both", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def format_table(res: Dict[str, Any]) -> str:
    """the rows of a pareto() result as text: one line per run, the fronts it is on at the end"""
    def f(v, spec):
        return "-" if v is None else format(v, spec)
    lines = [f"{'run':28s} {res['y']:>12s} {'ms':>9s} {'load':>5s} {'GMACs':>9s} {'ops':>10s}  front",
             "-" * 84]
    for r in sorted(res["rows"], key=lambda r: (r["ms"] is None, r["ms"] or 0.0)):
        on = [a for a in ("ms", "gmacs") if r.get("front_" + a)]
        lines.append(f"{r['run'][:28]:28s} {r['quality']:12.4f} {f(r['ms'], '9.3g'):>9s} {f(r['load1'], '5.1f'):>5s} "
                     f"{f(r['gmacs'], '9.4g'):>9s} {str(r['ops_method'] or '-'):>10s}  {', '.join(on) or '-'}")
    lines.append(f"quality on {res['n_common_frames']} common frames ({res['better']} is better); ms source: {res['ms_source']}")
    lines += [f"note: {n}" for n in res["notes"]]
    return "\n".join(lines)

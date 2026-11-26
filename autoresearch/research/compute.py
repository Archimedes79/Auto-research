"""The numbers of a verdict, computed where the metrics live: inside the experiment's snapshot, with the problem's interpreter (the judge starts it).

stdin: {"score": "<metric>.<key>", "guards": ["<metric>.<key>", ...], "boot": n,
        "control": {"<seed>": "<evaluation id>", ...}, "variant": {"<seed>": "<evaluation id>", ...}}
stdout: {"pooled": {control, variant, diff, lo, hi, n_scenes}, "seeds": {seed: {...}}, "guards": {name: {control, variant}}, "warnings": [...]}

diff = variant - control in the metric's own units. Pooled = the counts of all seeds of an arm put together per scene, then one paired
bootstrap over the scenes both arms have (the same draw for both)."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from Evaluation import Evaluation
from Evaluation.stats import bootstrap_scenes


def _split(name: str):
    metric, _, key = name.partition(".")
    if not key:
        raise SystemExit(f"a metric name is <metric>.<key>, got {name!r}")
    return metric, key


def main() -> int:
    q = json.load(sys.stdin)
    ev = Evaluation(project=os.getcwd())
    stores: Dict[str, Dict[str, Any]] = {"control": {}, "variant": {}}
    quant = None
    for arm in ("control", "variant"):
        for seed, eid in q[arm].items():
            info, store, qq = ev._open(eid)
            stores[arm][seed] = store
            quant = quant or qq
    metrics = quant._metrics()
    names = [q["score"]] + [g for g in q.get("guards", []) if g != q["score"]]
    by: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}               # metric -> arm -> scene -> counts of every seed
    for name in names:
        m, _ = _split(name)
        if m not in metrics:
            raise SystemExit(f"no metric {m!r} in this evaluation (has {sorted(metrics)})")
        if m in by:
            continue
        by[m] = {}
        for arm in ("control", "variant"):
            pooled: Dict[str, List[Any]] = {}
            for store in stores[arm].values():
                for scene, counts in store.by_scene(m).items():
                    pooled.setdefault(scene, []).extend(counts)
            by[m][arm] = pooled
    m, key = _split(q["score"])
    metric = metrics[m]
    better = {key: metric.direction(key)}
    red = lambda counts, keys=None: metric.reduce(counts, keys=keys)              # noqa: E731
    res = bootstrap_scenes(by[m]["control"], by[m]["variant"], red, [key], n=q.get("boot", 2000), seed=0, better=better)[key]
    out: Dict[str, Any] = {"pooled": {"control": res["a"], "variant": res["b"], "diff": res["diff"], "lo": res["lo"], "hi": res["hi"], "n_scenes": res["n_scenes"]},
                           "seeds": {}, "guards": {}, "warnings": []}
    for seed in sorted(set(stores["control"]) & set(stores["variant"]), key=int):
        a, b = stores["control"][seed].by_scene(m), stores["variant"][seed].by_scene(m)
        r = bootstrap_scenes(a, b, red, [key], n=q.get("boot", 2000), seed=0, better=better)[key]
        out["seeds"][seed] = {"control": r["a"], "variant": r["b"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"]}
    for g in q.get("guards", []):
        gm, gk = _split(g)
        red_g = metrics[gm].reduce
        out["guards"][g] = {arm: red_g([c for cs in by[gm][arm].values() for c in cs], keys=[gk])[gk] for arm in ("control", "variant")}
        out["guards"][g]["better"] = metrics[gm].direction(gk)
    if out["pooled"]["n_scenes"] < 30:
        out["warnings"].append(f"only {out['pooled']['n_scenes']} scenes: a scene bootstrap is too narrow below about 30 scenes")
    only = set(by[m]["control"]) ^ set(by[m]["variant"])
    if only:
        out["warnings"].append(f"{len(only)} scenes are in only one arm: the interval uses the {out['pooled']['n_scenes']} common ones")
    json.dump(out, sys.stdout, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

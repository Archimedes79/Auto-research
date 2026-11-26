"""ledger.csv of a problem: generated from the records (run.json, result.json, verdict.json of every experiment folder), never edited.

One row per run: experiment, run, arm, seed, status, score, the verdict of its arm, commit, dates. `research index <problem> --check` exits 1
when the file is not what the records give (a test of the records, e.g. before a commit)."""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List

from research.paths import problem_dir, read_json

COLUMNS = ("experiment", "run", "arm", "seed", "status", "score_metric", "score", "verdict", "reason", "commit", "started", "ended")


def rows(problem: str) -> List[Dict[str, Any]]:
    base = os.path.join(problem_dir(problem), "experiments")
    out: List[Dict[str, Any]] = []
    for exp in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        folder = os.path.join(base, exp)
        try:
            verdict = read_json(os.path.join(folder, "verdict.json"))
        except (OSError, ValueError):
            verdict = {}
        runs = os.path.join(folder, "runs")
        for rid in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
            try:
                run = read_json(os.path.join(runs, rid, "run.json"))
            except (OSError, ValueError):
                continue
            try:
                res = read_json(os.path.join(runs, rid, "result.json"))
            except (OSError, ValueError):
                res = {}
            arm = verdict.get("arms", {}).get(run.get("arm"), {})
            out.append({"experiment": run.get("experiment"), "run": rid, "arm": run.get("arm"), "seed": run.get("seed"), "status": run.get("status"),
                        "score_metric": res.get("score", {}).get("metric", ""), "score": res.get("score", {}).get("value", ""),
                        "verdict": arm.get("verdict", "control" if verdict and run.get("arm") == verdict.get("control") else ""),
                        "reason": arm.get("reason", ""), "commit": str(run.get("commit", ""))[:12], "started": run.get("started", ""), "ended": run.get("ended", "")})
    return out


def text(problem: str) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows(problem):
        w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})
    return buf.getvalue()


def path_of(problem: str) -> str:
    return os.path.join(problem_dir(problem), "ledger.csv")


def write(problem: str) -> str:
    path = path_of(problem)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text(problem))
    os.replace(tmp, path)
    return path


def current(problem: str) -> bool:
    try:
        with open(path_of(problem), encoding="utf-8", newline="") as f:
            return f.read() == text(problem)
    except OSError:
        return False

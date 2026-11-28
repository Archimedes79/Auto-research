"""ledger.csv of a problem: generated from the records (run.json, result.json, verdict.json of every experiment folder), never edited.

One row per run: experiment, run, arm, seed, status, score, the verdict of its arm, commit, dates, what scored it (evaluator / ground truth) and
whether the result is still comparable: a result counts as long as it was scored by the evaluator and against the ground truth of the newest
run of its split; after a change of either, every older row reads `no: ...` (its score and verdict stay as history and say nothing about today's
systems: measure them again). `research index <problem> --check` exits 1 when the file is not what the records give (a test of the records,
e.g. before a commit)."""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List

from research.paths import problem_dir, read_json

COLUMNS = ("experiment", "run", "arm", "seed", "status", "score_metric", "score", "verdict", "reason", "commit", "started", "ended", "scored_by", "comparable")


def _scored(res: Dict[str, Any]) -> Dict[str, str]:
    """what shaped a result's numbers: its split, the evaluator and the ground truth (name and content signature); {} without a result"""
    ev = res.get("evaluation") or {}
    if not res.get("score"):
        return {}
    return {"split": str(res.get("split")), "evaluator": str(ev.get("evaluator") or ""), "gt": str(ev.get("gt") or ""), "gt_signature": str(ev.get("gt_signature") or "")}


def comparability(mine: Dict[str, str], now: Dict[str, str]) -> str:
    """'yes' when a result was scored like the reference `now` (both from `_scored`, the same split), else 'no: ' and what differs. The ground
    truth is compared by its content signature where both records hold one, else by its name"""
    differs = []
    if mine["evaluator"] != now["evaluator"]:
        differs.append(f"evaluator {mine['evaluator'] or '?'} is not today's {now['evaluator'] or '?'}")
    key = "gt_signature" if mine["gt_signature"] and now["gt_signature"] else "gt"
    if mine[key] != now[key]:
        differs.append(f"ground truth {mine[key] or '?'} is not today's {now[key] or '?'}")
    return "yes" if not differs else "no: " + "; ".join(differs)


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
                        "reason": arm.get("reason", ""), "commit": str(run.get("commit", ""))[:12], "started": run.get("started", ""), "ended": run.get("ended", ""),
                        "_scored": _scored(res)})
    newest: Dict[str, Dict[str, str]] = {}                          # split -> what scored the run started last
    for r in sorted((r for r in out if r["_scored"]), key=lambda r: (str(r["started"]), str(r["run"]))):
        newest[r["_scored"]["split"]] = r["_scored"]
    for r in out:
        s = r.pop("_scored")
        r["scored_by"] = f"{s['evaluator']} / {s['gt_signature'] or s['gt']}" if s else ""
        r["comparable"] = comparability(s, newest[s["split"]]) if s else ""
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
    except FileNotFoundError:
        return not rows(problem)                 # nothing recorded yet needs no ledger (a new problem)
    except OSError:
        return False

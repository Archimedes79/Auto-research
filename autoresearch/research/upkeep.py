"""anchor, compare, clean: the upkeep of a problem's runs.

anchor   a finished run evaluated again with the code of the working tree, into a fresh folder: do its numbers still come out? (the champion's
         check after any change of the problem's code; research/anchor.py does the work with the problem's interpreter)
compare  two finished runs side by side: the paired scene bootstrap of the score and the guards (a diagnostic, never a verdict)
clean    the run folders that nothing needs any more: runs of judged experiments that are no champion and no comparison network (their
         records stay in git); listed with their sizes, deleted only with --delete under a decided decision (hard rule 5, D154). Folders
         that are no run of an experiment (imported networks, caches, the code snapshots of live runs) are never touched."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from research import gates
from research.launch import load_problem
from research.paths import RUN_ID, ResearchError, interpreter, module_path, problem_dir, read_json, record_dir, repo, runs_root


def _env(problem: str) -> Dict[str, str]:
    return dict(os.environ, RESEARCH_REPO=repo(), RESEARCH_RUNS=runs_root(problem), PYTHONPATH=module_path(repo(), problem))


def anchor(problem: str, rid: str, tolerance: Optional[float] = None) -> Dict[str, Any]:
    p = load_problem(problem)
    tol = p["evaluate"].get("anchor_tolerance", 0.0) if tolerance is None else tolerance
    r = subprocess.run([interpreter(p["python"]), "-m", "research.anchor", problem, rid, "--tolerance", str(tol)], cwd=problem_dir(problem),
                       env=_env(problem), capture_output=True)
    if r.returncode not in (0, 1):
        raise ResearchError(f"the anchor test of {rid} did not run: {r.stderr.decode('utf-8', 'replace')[-1500:]}")
    return json.loads(r.stdout.decode())


def compare(problem: str, a: str, b: str, boot: int = 2000) -> Dict[str, Any]:
    p = load_problem(problem)
    ids = {}
    for rid in (a, b):
        try:
            ids[rid] = read_json(os.path.join(record_dir(problem, rid), "result.json"))["evaluation"]["id"]
        except (OSError, KeyError, ValueError):
            raise ResearchError(f"{rid} has no result.json") from None
    q = {"score": p["score"]["metric"], "guards": [g["metric"] for g in p.get("guards", [])], "boot": boot, "control": {"0": ids[a]}, "variant": {"0": ids[b]}}
    r = subprocess.run([interpreter(p["python"]), "-m", "research.compute"], cwd=problem_dir(problem), env=_env(problem), input=json.dumps(q).encode(),
                       capture_output=True)
    if r.returncode != 0:
        raise ResearchError(f"the comparison failed: {r.stderr.decode('utf-8', 'replace')[-1500:]}")
    return {"control": a, "variant": b, "by": "research compare (a diagnostic, never a verdict)", **json.loads(r.stdout.decode())}


def _size(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def kept_runs(problem: str) -> List[str]:
    """the run folders that stay: every seed of the champion and the comparison networks"""
    try:
        idx = read_json(os.path.join(repo(), "champions", problem, "champions.json"))
    except (OSError, ValueError):
        return []
    keep = []
    for name in [idx.get("champion")] + list(idx.get("comparison", [])):
        if not name:
            continue
        try:
            facts = read_json(os.path.join(repo(), "champions", problem, name, "network.json"))
            keep += [s["run"] for s in facts["seeds"].values()]
        except (OSError, ValueError, KeyError):
            continue
    return keep


def clean(problem: str, delete: bool = False, decision: str = "") -> Dict[str, Any]:
    root = runs_root(problem)
    if not os.path.isfile(os.path.join(repo(), "champions", problem, "champions.json")):
        raise ResearchError(f"champions/{problem}/champions.json is missing: without it nothing is known to be kept, so nothing is listed for deletion")
    keep = set(kept_runs(problem))
    items = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not RUN_ID.match(name) or name in keep:
            continue
        exp = name.split(".")[0]
        try:
            folder = os.path.dirname(os.path.dirname(record_dir(problem, name)))
        except ResearchError:
            continue
        if not os.path.isfile(os.path.join(folder, "verdict.json")):
            continue                                                   # an experiment without a verdict still needs its runs
        items.append({"run": name, "experiment": exp, "bytes": _size(os.path.join(root, name))})
    out = {"runs": items, "bytes": sum(i["bytes"] for i in items), "kept": sorted(keep)}
    if delete:
        if decision not in gates.decided(problem):
            raise ResearchError(f"deleting run outputs needs a decided decision (got {decision!r}): list them first, the owner confirms the list")
        for i in items:
            shutil.rmtree(os.path.join(root, i["run"]))
        out["deleted"] = len(items)
    return out

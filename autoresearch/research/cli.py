"""The command line: python -m research <verb> ... (from autoresearch/, or with autoresearch/ on PYTHONPATH). Exit 0 done, 1 refused or failed,
2 wrong arguments. --json prints the result as json."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

from research import adopt as adopt_mod
from research import gates, index, judge, launch, worker
from research.paths import (RUN_ID, ResearchError, experiment_dir, machine, machine_file, parse_run, problem_dir, read_json, record_dir, rel,
                            run_dir, write_json)


def _runs(problem: str, target: str) -> List[str]:
    """the run ids a target names: one run, or every run of an experiment (other entries of its runs/ folder are no runs)"""
    if "." in target:
        parse_run(target)
        return [target]
    folder = os.path.join(experiment_dir(problem, target), "runs")
    return sorted(d for d in os.listdir(folder) if RUN_ID.match(d)) if os.path.isdir(folder) else []


def status(problem: str, target: str = "") -> List[Dict[str, Any]]:
    rows = []
    base = os.path.join(problem_dir(problem), "experiments")
    exps = [target.split(".")[0]] if target else sorted({d.split("_")[0] for d in os.listdir(base) if re.match(r"^E\d+(_|$)", d)} if os.path.isdir(base) else [])
    for exp in exps:
        for rid in _runs(problem, target if target and "." in target else exp):
            try:
                run = read_json(os.path.join(record_dir(problem, rid), "run.json"))
            except (OSError, ValueError):
                continue
            beat, last = "", ""
            try:
                t, _, what = open(os.path.join(run_dir(problem, rid), "heartbeat"), encoding="utf-8").read().partition(" ")
                beat = f"{time.time() - float(t):.0f} s ago ({what.strip()})"
            except (OSError, ValueError, ResearchError):
                pass
            try:
                with open(os.path.join(run_dir(problem, rid), "log.jsonl"), encoding="utf-8") as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
                row = json.loads(lines[-1]) if lines else {}
                last = f"step {row.get('step')} " + " ".join(f"{k} {v:.4g}" for k, v in row.items() if k not in ("step",) and isinstance(v, (int, float)))[:120]
            except (OSError, ValueError, ResearchError):
                pass
            steps = " ".join(f"{k}:{v.get('status')}" for k, v in run.get("steps", {}).items())
            rows.append({"run": rid, "status": run.get("status"), "steps": steps, "heartbeat": beat, "last": last})
    return rows


def stop(problem: str, target: str) -> List[str]:
    done = []
    for rid in _runs(problem, target):
        run = read_json(os.path.join(record_dir(problem, rid), "run.json"))
        if run.get("status") in ("running", "queued"):
            os.makedirs(run_dir(problem, rid), exist_ok=True)
            write_json(os.path.join(run_dir(problem, rid), "STOP_REQUESTED"), {"at": time.strftime("%Y-%m-%d %H:%M:%S")})
            done.append(rid)
    return done


def _repo_path(top: str, path: str) -> str:
    """a path for `commit`, relative to the top of the repository: as git reads it (relative to here), else relative to the top (the
    commands run from autoresearch/, the records are named from the top); a path that is in neither place is left to commit_paths to refuse"""
    here = os.path.relpath(os.path.abspath(path), top).replace("\\", "/")
    if os.path.exists(os.path.join(top, here)) or os.path.isabs(path):
        return here
    from_top = os.path.normpath(path).replace("\\", "/")
    return from_top if os.path.exists(os.path.join(top, from_top)) else here


STALE_S = 600                                                  # a heartbeat older than this: the worker is gone


def heartbeat_age(problem: str, rid: str) -> float:
    try:
        with open(os.path.join(run_dir(problem, rid), "heartbeat"), encoding="utf-8") as f:
            return time.time() - float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return float("inf")


def resume(problem: str, target: str, failed: bool = False) -> List[str]:
    """start the worker again for stopped runs (and failed ones with --failed); finished steps are skipped, a training resumes from its checkpoint"""
    if gates.hold_refusal(problem):
        raise ResearchError(gates.hold_refusal(problem))
    done = []
    for rid in _runs(problem, target):
        path = os.path.join(record_dir(problem, rid), "run.json")
        run = read_json(path)
        if worker.step_alive(problem, rid):
            raise ResearchError(f"{rid}: a step of it still runs (pid in {os.path.join(run_dir(problem, rid), 'step.pid')}): stop it first")
        orphan = run.get("status") == "running" and heartbeat_age(problem, rid) > STALE_S       # its worker died (a reboot, a killed WSL)
        if run.get("status") in ("stopped", "queued") or orphan or (failed and run.get("status") == "failed"):
            run.update(status="queued", resumed=time.strftime("%Y-%m-%d %H:%M:%S"))      # not a final state any more: the worker takes it from here
            write_json(path, run)
            launch.spawn_worker(problem, rid)
            done.append(rid)
    return done


def machine_init(data: str, runs: str, python: List[str], gpu_slots: int) -> str:
    path = machine_file()
    if os.path.exists(path):
        raise ResearchError(f"{path} exists: edit it by hand")
    envs = dict(x.split("=", 1) for x in python)
    write_json(path, {"data": os.path.abspath(data), "runs": os.path.abspath(runs), "python": envs, "gpu_slots": gpu_slots})
    machine()
    return path


def build() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="research", description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the result as json")
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("new", help="a new experiment folder with the card template")
    s.add_argument("problem"), s.add_argument("slug"), s.add_argument("title")
    for verb, text in (("check", "everything start would refuse"), ("start", "snapshot and start the runs of an experiment")):
        s = sub.add_parser(verb, help=text)
        s.add_argument("problem"), s.add_argument("experiment")
        if verb == "start":
            s.add_argument("--arms", default=""), s.add_argument("--seeds", default="")
    s = sub.add_parser("status", help="the runs and their state")
    s.add_argument("problem"), s.add_argument("target", nargs="?", default="")
    for verb, text in (("stop", "ask runs to stop at their next checkpoint"), ("resume", "continue stopped runs")):
        s = sub.add_parser(verb, help=text)
        s.add_argument("problem"), s.add_argument("target")
        if verb == "resume":
            s.add_argument("--failed", action="store_true", help="also failed runs")
    s = sub.add_parser("judge", help="the verdict by the snapshot's rule")
    s.add_argument("problem"), s.add_argument("experiment")
    s.add_argument("--write", action="store_true", help="write verdict.json and the ledger and commit the records")
    s.add_argument("--no-commit", action="store_true"), s.add_argument("--boot", type=int, default=2000)
    s = sub.add_parser("adopt", help="a KEEP arm becomes the champion (the owner's decision)")
    s.add_argument("problem"), s.add_argument("experiment"), s.add_argument("arm"), s.add_argument("--decision", required=True)
    s.add_argument("--by-hand", default=""), s.add_argument("--tolerance", type=float, default=None, help="default: problem.json evaluate.anchor_tolerance"), s.add_argument("--no-commit", action="store_true")
    s = sub.add_parser("register", help="a measured network as champion or comparison network (the owner's decision)")
    s.add_argument("problem"), s.add_argument("experiment"), s.add_argument("arm"), s.add_argument("--decision", required=True)
    s.add_argument("--role", choices=("champion", "comparison"), required=True), s.add_argument("--no-commit", action="store_true")
    s = sub.add_parser("test", help="evaluate a finished run once on the holdout split (a TEST log row under a decided decision)")
    s.add_argument("problem"), s.add_argument("run")
    s = sub.add_parser("report", help="build a report again from the records and the stored predictions (into the run root)")
    s.add_argument("problem"), s.add_argument("folder"), s.add_argument("--strict", action="store_true")
    s = sub.add_parser("tests", help="the problem's own tests (problem.json \"tests\") with its interpreter and module path; exit 1 when one fails")
    s.add_argument("problem"), s.add_argument("--only", default="", help="only the tests whose path contains this text")
    s = sub.add_parser("anchor", help="evaluate a finished run again with the working tree's code: do its numbers still come out?")
    s.add_argument("problem"), s.add_argument("run"), s.add_argument("--tolerance", type=float, default=None)
    s = sub.add_parser("compare", help="two finished runs side by side (a diagnostic, never a verdict)")
    s.add_argument("problem"), s.add_argument("control"), s.add_argument("variant"), s.add_argument("--boot", type=int, default=2000)
    s = sub.add_parser("clean", help="list the run folders nothing needs any more into a list file in the run root; --delete <list file> "
                                     "--decision D<n> deletes the runs of that list that are still deletable")
    s.add_argument("problem"), s.add_argument("--delete", default="", metavar="LIST", help="the list file the owner confirmed")
    s.add_argument("--decision", default="", help="the decided decision whose row names the list's hash (the owner saw that list)")
    s = sub.add_parser("index", help="write ledger.csv from the records")
    s.add_argument("problem"), s.add_argument("--check", action="store_true", help="exit 1 when ledger.csv is not what the records give")
    s = sub.add_parser("hold", help="no start or resume of a problem until released")
    s.add_argument("problem"), s.add_argument("reason")
    s = sub.add_parser("release")
    s.add_argument("problem")
    s = sub.add_parser("grant", help="allow edits of a protected path for some hours under a decided decision")
    s.add_argument("path"), s.add_argument("--decision", required=True), s.add_argument("--hours", type=float, default=2.0), s.add_argument("--note", required=True)
    s = sub.add_parser("machine", help="show the machine file, or write it (init)")
    s.add_argument("action", nargs="?", default="show", choices=("show", "init"))
    s.add_argument("--data", default=""), s.add_argument("--runs", default=""), s.add_argument("--python", action="append", default=[])
    s.add_argument("--gpu-slots", type=int, default=2)
    s = sub.add_parser("commit", help="commit the working-tree state of these paths only (what other chats edit or stage stays out)")
    s.add_argument("paths", nargs="+"), s.add_argument("-m", "--message", required=True)
    s = sub.add_parser("push-check", help="the commits not yet pushed and what in them never belongs into git (exit 1 when something does)")
    s.add_argument("--upstream", default="@{upstream}")
    s = sub.add_parser("worker", help="(internal) run the steps of one run")
    s.add_argument("problem"), s.add_argument("run"), s.add_argument("--test", action="store_true")
    return ap


def run(a: argparse.Namespace) -> Any:
    v = a.verb
    if v == "new":
        return {"folder": rel(launch.new(a.problem, a.title, a.slug))}
    if v == "check":
        return launch.check(a.problem, a.experiment)
    if v == "start":
        return launch.start(a.problem, a.experiment, [x for x in a.arms.split(",") if x] or None, [int(x) for x in a.seeds.split(",") if x] or None)
    if v == "status":
        return status(a.problem, a.target)
    if v == "stop":
        return {"stop requested": stop(a.problem, a.target)}
    if v == "resume":
        return {"resumed": resume(a.problem, a.target, a.failed)}
    if v == "judge":
        res = judge.judge(a.problem, a.experiment, a.boot)
        if a.write and res["verdict"] == "DONE":
            res["records_commit"] = judge.write(a.problem, a.experiment, res, commit=not a.no_commit)
        return res
    if v == "adopt":
        return adopt_mod.adopt(a.problem, a.experiment, a.arm, a.decision, a.by_hand, a.tolerance, commit=not a.no_commit)
    if v == "register":
        return adopt_mod.register(a.problem, a.experiment, a.arm, a.decision, a.role, commit=not a.no_commit)
    if v == "test":
        return launch.test(a.problem, a.run)
    if v == "report":
        from research import report
        return report.build(a.problem, a.folder, a.strict)
    if v in ("tests", "anchor", "compare", "clean"):
        from research import upkeep
        if v == "tests":
            res = upkeep.tests(a.problem, a.only)
            if not res["passed"]:
                raise ResearchError("failed:\n" + "\n".join(f"  {r['test']} (exit {r['exit']}): {r['last']}" for r in res["tests"] if not r["passed"]))
            return res
        if v == "anchor":
            res = upkeep.anchor(a.problem, a.run, a.tolerance)
            if not res["agree"]:
                raise ResearchError("the numbers do not come out:\n" + json.dumps(res["differ"], indent=1, default=str))
            return res
        if v == "compare":
            return upkeep.compare(a.problem, a.control, a.variant, a.boot)
        return upkeep.clean(a.problem, a.delete, a.decision)
    if v == "commit":
        from research import gitio
        trailer = "\n\nCo-Authored-By: " + os.environ["RESEARCH_COAUTHOR"] if os.environ.get("RESEARCH_COAUTHOR") else ""
        top = gitio.git(os.getcwd(), "rev-parse", "--show-toplevel")
        c = gitio.commit_paths(top, [_repo_path(top, p) for p in a.paths], a.message + trailer)
        return {"commit": c or "nothing changed"}
    if v == "push-check":
        from research import gitio
        res = gitio.push_problems(gitio.git(os.getcwd(), "rev-parse", "--show-toplevel"), a.upstream)
        if res["forbidden"]:
            raise ResearchError("never into git:\n  " + "\n  ".join(res["forbidden"]))
        return res
    if v == "index":
        if a.check:
            ok = index.current(a.problem)
            if not ok:
                raise ResearchError(f"{rel(index.path_of(a.problem))} is not what the records give: `research index {a.problem}`")
            return {"current": True}
        return {"written": rel(index.write(a.problem))}
    if v == "hold":
        gates.hold(a.problem, a.reason)
        return {"held": a.problem}
    if v == "release":
        return {"released": gates.release(a.problem)}
    if v == "grant":
        path = a.path.replace("\\", "/")
        prob = path.split("/")[1] if path.startswith("problems/") and path.count("/") >= 2 else None
        return gates.grant(path, a.decision, a.hours, a.note, prob)
    if v == "machine":
        if a.action == "init":
            return {"written": machine_init(a.data, a.runs, a.python, a.gpu_slots)}
        return {"file": machine_file(), **machine()}
    if v == "worker":
        sys.exit(worker.main(a.problem, a.run, a.test))
    raise ResearchError(f"unknown verb {v}")


def show(res: Any) -> str:
    if isinstance(res, list):
        return "\n".join("  ".join(str(v) for v in r.values()) if isinstance(r, dict) else str(r) for r in res) or "(nothing)"
    return json.dumps(res, indent=1, default=str)


def main(argv: List[str]) -> int:
    a = build().parse_args(argv)
    try:
        res = run(a)
    except ResearchError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1, default=str) if a.json else show(res))
    return 0

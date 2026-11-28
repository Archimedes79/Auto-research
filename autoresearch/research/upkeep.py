"""tests, py, anchor, compare, clean: the upkeep of a problem's code and runs.

tests    the problem's own tests (problem.json "tests": modules of the problem, started with -P -m) and, when it has a
         training/architecture.json, the drawing check against the code (reporter/archdraw.py check), each with the problem's interpreter
         and module path (paths.module_path of the working tree) from its folder, in a session of its own; passed =
         exit 0 and no "Traceback (most recent call last)" in the output (a script can die with exit 0 behind a wrapper, so a test that prints
         an expected traceback fails); past the timeout the test and all it started are killed; after any change of the problem's code
py       python by hand as a test gets it: the problem's interpreter and the module path of the working tree, from the caller's folder;
         with -P when the problem is a package, without when it is none yet (a script then finds the modules next to it, as in `tests`)
anchor  a finished run evaluated again with the code of the working tree, into a fresh folder: do its numbers still come out? (the champion's
         check after any change of the problem's code; research/anchor.py does the work with the problem's interpreter)
compare  two finished runs side by side: the paired scene bootstrap of the score and the guards (a diagnostic, never a verdict)
clean    the run folders that nothing needs any more (hard rule 5, D154; their records stay in git). `clean <p>` writes the list (run ids,
         sizes, the time, a hash) into a list file in the problem's run root; the owner confirms exactly that list with a decided row that
         names its hash (the first 12 hex), and `clean <p> --delete <list file> --decision D<n>` checks the whole list, then deletes its runs,
         each checked again just before. A run stays while it is not finished (ok, failed, killed), its step lives, its experiment has no
         verdict or an arm KEEP, INCONCLUSIVE, TIE or MEASURED (an adoption, a registration or another seed needs it), a card names it (`from`,
         `needs`, or a `{runs}/<run>` path in its settings), or it is a seed of a champion or comparison network; an unreadable champions.json
         or network.json, or a card that cannot be read, refuses. Folders that are no run (imported networks, caches, the code snapshots) are
         never touched."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

from research import gates, spec
from research.launch import load_problem
from research.paths import (CORE, RUN_ID, ResearchError, child_env, core_step, interpreter, is_package, module_path, parse_run, problem_dir,
                            read_json, record_dir, rel, repo, runs_root, write_json)
from research.worker import step_alive

DRAWING = "training/architecture.json"                                  # a problem's system drawing (reporter/archdraw.py)


def _env(problem: str) -> Dict[str, str]:
    return child_env(RESEARCH_REPO=repo(), RESEARCH_RUNS=runs_root(problem), PYTHONPATH=module_path(repo(), problem))


def tests(problem: str, only: str = "", timeout: float = 3600) -> Dict[str, Any]:
    p = load_problem(problem)
    checks = [(t, t.split()) for t in p.get("tests", [])]                # name -> the arguments of the interpreter
    if os.path.isfile(os.path.join(problem_dir(problem), DRAWING)):
        checks.append((f"archdraw check {DRAWING}", [os.path.join(CORE, "reporter", "archdraw.py"), "check", DRAWING]))
    checks = [(name, args) for name, args in checks if only in name]
    if not checks:
        raise ResearchError(f"problem.json of {problem} lists no test{' matching ' + repr(only) if only else ''} (key \"tests\")")
    rows = []
    for name, args in checks:
        t0 = time.time()
        start = [] if args[0].endswith(".py") else ["-P", "-m"]     # a module of the problem; a script: the drawing check, a problem that is no package yet
        proc = subprocess.Popen([interpreter(p["python"]), *start, *args], cwd=problem_dir(problem), env=_env(problem), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, code = proc.communicate(timeout=timeout)[0].decode("utf-8", "replace"), proc.returncode
        except subprocess.TimeoutExpired:                            # the test and everything it started (trainers, loaders) end
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            out, code = f"timed out after {timeout:.0f} s: its process group was killed", -1
        lines = [ln for ln in out.splitlines() if ln.strip()]
        rows.append({"test": name, "passed": code == 0 and "Traceback (most recent call last)" not in out, "exit": code,
                     "seconds": round(time.time() - t0, 1), "last": lines[-1][:200] if lines else ""})
    return {"problem": problem, "passed": all(r["passed"] for r in rows), "tests": rows}


def py(problem: str, args: List[str]) -> int:
    """work by hand: the problem's interpreter with the module path of the working tree (what a test gets), from the caller's folder ->
    its exit code. A package starts with -P (the given roots only); a problem that is none yet starts as its tests do, so a script still
    finds the modules next to it. A diagnostic by definition: no RESEARCH_RUN reaches it, so a holdout split stays closed and no record
    is written"""
    start = ["-P"] if is_package(repo(), problem) else []
    return subprocess.call([interpreter(load_problem(problem)["python"]), *start, *args], env=_env(problem))


def anchor(problem: str, rid: str, tolerance: Optional[float] = None) -> Dict[str, Any]:
    p = load_problem(problem)
    tol = p["evaluate"].get("anchor_tolerance", 0.0) if tolerance is None else tolerance
    r = subprocess.run(core_step(p["python"], "research.anchor", problem, rid, "--tolerance", str(tol)), cwd=problem_dir(problem),
                       env=_env(problem), capture_output=True)
    lines = [ln for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
    try:
        res = json.loads(lines[-1])                                  # research/anchor.py answers with one json line, also when it fails
    except (IndexError, ValueError):
        res = None
    if r.returncode not in (0, 1) or not isinstance(res, dict) or "agree" not in res:
        why = res.get("error") if isinstance(res, dict) else ""
        raise ResearchError(f"the anchor test of {rid} did not run: {why or r.stderr.decode('utf-8', 'replace')[-1500:]}")
    return res


def compare(problem: str, a: str, b: str, boot: int = 2000) -> Dict[str, Any]:
    p = load_problem(problem)
    ids = {}
    for rid in (a, b):
        try:
            ids[rid] = read_json(os.path.join(record_dir(problem, rid), "result.json"))["evaluation"]["id"]
        except (OSError, KeyError, ValueError):
            raise ResearchError(f"{rid} has no result.json") from None
    q = {"score": p["score"]["metric"], "guards": [g["metric"] for g in p.get("guards", [])], "boot": boot, "control": {"0": ids[a]}, "variant": {"0": ids[b]}}
    r = subprocess.run(core_step(p["python"], "research.compute"), cwd=problem_dir(problem), env=_env(problem), input=json.dumps(q).encode(),
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
    """every seed of the champion and the comparison networks; a champions.json or a network.json it names that cannot be read refuses
    (a problem with nothing registered yet, no champions.json and no network.json, keeps none)"""
    base = os.path.join(repo(), "champions", problem)
    registered = os.path.isdir(base) and any(os.path.isfile(os.path.join(base, d, "network.json")) for d in os.listdir(base))
    if not os.path.exists(os.path.join(base, "champions.json")) and not registered:
        return []
    try:
        idx = read_json(os.path.join(base, "champions.json"))
        names = [n for n in [idx.get("champion")] + list(idx.get("comparison", [])) if n]
        return [s["run"] for n in names for s in read_json(os.path.join(base, n, "network.json"))["seeds"].values()]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ResearchError(f"champions/{problem} cannot be read ({type(e).__name__}: {e}): without it nothing is known to be kept, so nothing "
                            "is listed or deleted") from None


RUN_PATH = re.compile(r"\{runs\}/(E\d+\.[a-z][a-z0-9]*_s\d+)")


def referenced(problem: str) -> Dict[str, str]:
    """E<n>.<arm> (`from`, `needs`) or a run id (a `{runs}/<run>` path in an arm's settings) -> the experiment folder whose card names it
    (the working cards: a card not yet started counts too)"""
    base = os.path.join(problem_dir(problem), "experiments")
    refs: Dict[str, str] = {}
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        path = os.path.join(base, d, "experiment.md")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                card = spec.parse_card(f.read(), rel(path))
        except (OSError, ResearchError) as e:
            raise ResearchError(f"{e}: without the card it is unknown which runs it builds on, so nothing is listed or deleted") from None
        arms = card.get("arms") if isinstance(card.get("arms"), dict) else {}
        for arm in (a for a in arms.values() if isinstance(a, dict)):
            for ref in [arm.get("from")] + list((arm.get("needs") or {}).values()) + RUN_PATH.findall(json.dumps(arm.get("set", {}))):
                if isinstance(ref, str):
                    refs[ref] = d
    return refs


FINISHED = ("ok", "failed", "killed")
OPEN = ("KEEP", "INCONCLUSIVE", "TIE", "MEASURED")                     # arms whose runs an adoption (TIE: by hand), a registration or a seed needs


def why_kept(problem: str, rid: str, keep: List[str], refs: Dict[str, str]) -> str:
    """'' when the folder of this run may be deleted, else why it stays (the rule of clean, in one place)"""
    exp, arm, _ = parse_run(rid)
    if rid in keep:
        return "a seed of a champion or comparison network"
    for ref in (rid, f"{exp}.{arm}"):
        if ref in refs:
            return f"the card of {refs[ref]} names {ref} (from, needs or a {{runs}} path)"
    try:
        rec = record_dir(problem, rid)
        status = read_json(os.path.join(rec, "run.json")).get("status")
    except (OSError, ValueError, ResearchError):
        return "no readable record (experiment folder, run.json)"
    if status not in FINISHED:
        return f"not finished ({status})"
    if step_alive(problem, rid):
        return "its step still runs (step.pid)"
    try:
        verdict = read_json(os.path.join(os.path.dirname(os.path.dirname(rec)), "verdict.json"))
    except OSError:
        return f"{exp} has no verdict"
    except ValueError:
        return f"the verdict.json of {exp} cannot be read"
    open_arms = [f"{a} {v.get('verdict')}" for a, v in sorted(verdict.get("arms", {}).items()) if isinstance(v, dict) and v.get("verdict") in OPEN]
    return f"the verdict of {exp} has {', '.join(open_arms)}: an adoption, a registration or another seed needs its runs" if open_arms else ""


def _list_hash(lst: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: lst.get(k) for k in ("problem", "made", "runs", "bytes")}, sort_keys=True).encode()).hexdigest()[:16]


def clean(problem: str, delete: str = "", decision: str = "") -> Dict[str, Any]:
    """without `delete`: list the deletable runs into a list file; with it (the list file's path) and the decided decision that confirms
    that list: delete them"""
    lst = _confirmed(problem, delete, decision) if delete else None
    keep, refs = kept_runs(problem), referenced(problem)
    return _delete(problem, delete, lst, decision, keep, refs) if lst is not None else _list(problem, keep, refs)


def confirm_text(lst: Dict[str, Any]) -> str:
    """a decided row that confirms this list (the hash is what is checked)"""
    size = sum(i["bytes"] for i in lst["runs"]) / 1e9
    return f"| <date> | D<n>: delete the runs of clean list {lst['hash'][:12]} ({len(lst['runs'])} runs, {size:.1f} GB) |"


def _confirmed(problem: str, list_path: str, decision: str) -> Dict[str, Any]:
    """the list file, checked as a whole before anything is deleted: written by `clean` and unchanged, every entry a run id, and the decided
    row of `decision` names its hash (so the owner saw this list, and a list written by hand names no row)"""
    try:
        lst = read_json(list_path)
    except (OSError, ValueError) as e:
        raise ResearchError(f"the list {list_path}: {e}") from None
    if not isinstance(lst, dict) or lst.get("problem") != problem or not isinstance(lst.get("runs"), list) or lst.get("hash") != _list_hash(lst):
        raise ResearchError(f"{list_path} is no list of `research clean {problem}`, or it was changed after the listing")
    bad = [i for i in lst["runs"] if not isinstance(i, dict) or not RUN_ID.match(str(i.get("run", ""))) or not isinstance(i.get("bytes"), int)]
    if bad:
        raise ResearchError(f"{list_path} holds entries that are no runs ({bad[:3]}): nothing is deleted")
    row = gates.decided_rows(problem).get(decision)
    if row is None or lst["hash"][:12] not in row.lower():
        raise ResearchError(f"{decision!r} is {'not a decided decision' if row is None else 'decided, but its row does not name this list'}: the owner "
                            f"confirms the list with a decided row that holds its hash, e.g. `{confirm_text(lst)}`")
    return lst


def _list(problem: str, keep: List[str], refs: Dict[str, str]) -> Dict[str, Any]:
    root = runs_root(problem)
    items, kept = [], {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if RUN_ID.match(name) and os.path.isdir(os.path.join(root, name)):
            why = why_kept(problem, name, keep, refs)
            if why:
                kept[name] = why
            else:
                items.append({"run": name, "bytes": _size(os.path.join(root, name))})
    lst = {"problem": problem, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": items, "bytes": sum(i["bytes"] for i in items)}
    lst["hash"] = _list_hash(lst)
    path = os.path.join(root, f"clean_{time.strftime('%Y%m%d_%H%M%S')}.json") if items else ""
    if path:
        write_json(path, lst)
    return {"list": path, **lst, "kept": kept, **({"confirm": f"the owner's decided row names the hash: {confirm_text(lst)}"} if path else {})}


def _delete(problem: str, list_path: str, lst: Dict[str, Any], decision: str, keep: List[str], refs: Dict[str, str]) -> Dict[str, Any]:
    root = runs_root(problem)
    deleted, kept, gone = [], {}, []
    for item in lst["runs"]:
        rid = item["run"]
        why = why_kept(problem, rid, keep, refs)                          # again, just before: a run may have been resumed or named since
        if not os.path.isdir(os.path.join(root, rid)):
            gone.append(rid)
        elif why:
            kept[rid] = why
        else:
            shutil.rmtree(os.path.join(root, rid))
            deleted.append(rid)
    return {"list": list_path, "decision": decision, "deleted": deleted, "kept": kept, "gone": gone}

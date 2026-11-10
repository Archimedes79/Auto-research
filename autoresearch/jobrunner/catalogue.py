"""The catalogue of jobs: which `Class.method` of which suite is a job, what it takes from the machine, which data it reads and writes.

A suite needs NO code change to be a job: the built-in table below names the four known methods, a project adds or overrides entries in its project.json
(`"jobrunner": {"catalogue": {"Toy.work": {...}}}`), and a suite may export its own table as a module attribute `JOBS` (same format) that wins over the built-in one
- that is the whole interface a suite has to the job runner.

An entry (every key but `suite`, `cls`, `method` and `how` is optional):
    suite   the module that holds the class (`Evaluation`; found on sys.path, the problem folder is on it too)
    cls     the class (built as `cls(project_folder)`, the convention of all suites), method  the method that does the work
    how     "process"  the runner starts `python -m jobrunner call ...` under run.py (slots, RAM gate, lock, ledger row): for long or GPU methods
            "launch"   the method itself starts a run.py run and returns (Training.run): the runner calls it in its own process
    may_feed_verdict  True: a `card` in the job registers its rule; without a card the run is a diagnostic (run.py --no-card) and never feeds a verdict
    needs   what the job takes: gpu (bool), gpu_max_used_mb (start only when the GPU memory in use is below this: run.py --gpu-max-mb), ram_gb (run.py --ram-mb,
            GPU jobs only), cpu (threads, recorded, no gate yet), est_hours (recorded)
    reads, writes  data references as templates over the arguments (`run:{network}`, `set:{frames}`, `eval:{network}@{frames}`): what the job needs and leaves
    python  the interpreter of the child process (default: project.json jobrunner.python, else the one of the runner)
    id_from  where a missing job id comes from ("job.id": args["job"]["id"])
    after_from  where the run ids it waits for stand in the arguments ("job.after": args["job"]["after"], a list or one id): for a `launch` call, whose own run.py
            start takes them, they are checked and shown by `check` (a typo is found before the start)
    splits  the arguments that name a split or a set of frames (default: split, frames, set, sets, subset): only their values are scanned for the holdout (TEST) split, never free text
    dry_run  the method of the same class that looks at the same arguments and starts nothing (`Evaluation.check`, `Runner.plan`): `check --deep` runs it in the interpreter of the job
            (the suite's own rules: that a network name is a run folder, that the data and the checkpoint open) with the arguments it knows, and reports its problems
Standard library only."""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

NEEDS_KEYS = ("gpu", "gpu_max_used_mb", "ram_gb", "cpu", "est_hours")
ENTRY_KEYS = ("suite", "cls", "method", "how", "may_feed_verdict", "needs", "reads", "writes", "python", "id_from", "after_from", "splits", "dry_run")
HOW = ("process", "launch")

BUILTIN: Dict[str, Dict[str, Any]] = {
    "Training.run": dict(suite="Training", cls="Training", method="run", how="launch", may_feed_verdict=True, needs=dict(gpu=True, gpu_max_used_mb=3000, ram_gb=10),
                         reads=["path:{job[config]}"], writes=["run:{job[id]}"], id_from="job.id", after_from="job.after", dry_run="check"),
    "Evaluation.run": dict(suite="Evaluation", cls="Evaluation", method="run", how="process", may_feed_verdict=True, needs=dict(gpu=True, gpu_max_used_mb=8000, ram_gb=6),
                           reads=["run:{network}", "set:{frames}"], writes=["eval:{network}@{frames}"], splits=["frames"], dry_run="check"),
    "Evaluation.score": dict(suite="Evaluation", cls="Evaluation", method="score", how="process", may_feed_verdict=True, needs=dict(cpu=2, ram_gb=4),
                             reads=["results:{results}", "set:{frames}"], writes=["scores:{results}@{frames}"], splits=["frames"]),
    "Runner.predict": dict(suite="runner", cls="Runner", method="predict", how="process", needs=dict(gpu=True, gpu_max_used_mb=8000, ram_gb=6),
                           reads=["run:{network}", "set:{frames}"], writes=["predictions:{network}@{frames}"], splits=["frames"], dry_run="plan"),
}


def check_entry(call: str, e: Any) -> List[str]:
    """the problems of one catalogue entry (empty: fine)"""
    if not isinstance(e, dict):
        return [f"{call}: the entry is not a dict"]
    bad = [f"{call}: unknown key {k!r}" for k in e if k not in ENTRY_KEYS and not str(k).startswith("_")]
    bad += [f"{call}: missing {k!r}" for k in ("suite", "cls", "method", "how") if not e.get(k)]
    if e.get("how") and e["how"] not in HOW:
        bad.append(f"{call}: how must be one of {HOW}, not {e['how']!r}")
    if call != f"{e.get('cls')}.{e.get('method')}":
        bad.append(f"{call}: the name must be cls.method ({e.get('cls')}.{e.get('method')})")
    needs = e.get("needs", {})
    if not isinstance(needs, dict) or any(k not in NEEDS_KEYS for k in needs):
        bad.append(f"{call}: needs must be a dict with keys from {NEEDS_KEYS}")
    for k in ("reads", "writes"):
        if k in e and not (isinstance(e[k], list) and all(isinstance(x, str) and ":" in x for x in e[k])):
            bad.append(f"{call}: {k} must be a list of 'kind:template' texts")
    if "splits" in e and not (isinstance(e["splits"], list) and all(isinstance(x, str) for x in e["splits"])):
        bad.append(f"{call}: splits must be a list of argument names")
    if "dry_run" in e and not (isinstance(e["dry_run"], str) and e["dry_run"] and e["dry_run"] != e.get("method")):
        bad.append(f"{call}: dry_run must be the name of another method of the class")
    return bad


@contextmanager
def import_paths(*dirs: str):
    """the folders put first on sys.path for the time of an import (autoresearch/ for the suites, the problem folder for the project's own packages)"""
    added = [d for d in dirs if d and d not in sys.path]
    for d in reversed(added):
        sys.path.insert(0, d)
    try:
        yield
    finally:
        for d in added:
            if d in sys.path:
                sys.path.remove(d)


def check_needs(needs: Dict[str, Any]) -> List[str]:
    """the problems of a `needs` dict: gpu is true / false, the others are numbers (a text like "6" is not one: it would reach run.py as a huge number)"""
    out = []
    for k, v in needs.items():
        if k == "gpu":
            if not isinstance(v, bool):
                out.append("needs.gpu must be true or false")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 <= v <= 1e6) or (k in ("gpu_max_used_mb", "cpu") and int(v) != v):
            out.append(f"needs.{k} must be a number" + (" (whole)" if k in ("gpu_max_used_mb", "cpu") else "") + " between 0 and 1e6")
    return out


def entries(cfg: Dict[str, Any], dirs: Tuple[str, ...] = ()) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """-> (the merged catalogue, problems). Order: built-in, the JOBS table of each suite that can be imported (the built-in suites, those named in project.json
    jobrunner.suites and those of the project's own entries), the project's own entries (last word)"""
    table = copy.deepcopy(BUILTIN)
    problems: List[str] = []
    mine = ((cfg.get("jobrunner") or {}).get("catalogue") or {})
    suites = {e["suite"] for e in table.values()} | set((cfg.get("jobrunner") or {}).get("suites") or []) | {e["suite"] for e in mine.values() if isinstance(e, dict) and e.get("suite")}
    for suite in sorted(suites):
        try:
            with import_paths(*dirs):
                jobs = getattr(importlib.import_module(suite), "JOBS", None)
        except (Exception, SystemExit):                                    # a suite that cannot be imported here (no torch, not in this project, exits on import) keeps its other entries
            jobs = None
        for call, e in (jobs or {}).items():
            table[call] = copy.deepcopy(e)
    for call, e in mine.items():
        table[call] = {**table.get(call, {}), **copy.deepcopy(e)} if isinstance(e, dict) else e
    for call, e in table.items():
        problems += check_entry(call, e)
    return table, problems


_SNIPPET = """import sys, json, importlib, inspect, contextlib
dirs, suite, cls, method = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path[:0] = dirs
with contextlib.redirect_stdout(sys.stderr):
    fn = getattr(getattr(importlib.import_module(suite), cls), method)
out = []
for k, p in inspect.signature(fn).parameters.items():
    if k == 'self':
        continue
    var = p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    d = None if p.default is inspect.Parameter.empty else p.default
    try:
        json.dumps(d)
    except TypeError:
        d = repr(d)
    out.append(dict(name=p.name, required=p.default is inspect.Parameter.empty and not var, default=d, var=var))
print(json.dumps(out))
"""


def signature(entry: Dict[str, Any], dirs: Tuple[str, ...] = (), python: str = "") -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """-> ([{name, required, default}] of the method, '') or (None, why it could not be read). The class is imported here; when that fails (no torch in this
    interpreter) and `python` is another interpreter, the same is asked of that one in a child process (the one the job will run in)."""
    try:
        with import_paths(*dirs):
            cls = getattr(importlib.import_module(entry["suite"]), entry["cls"])
        fn = getattr(cls, entry["method"])
        params = [p for k, p in inspect.signature(fn).parameters.items() if k != "self"]
        return [dict(name=p.name, required=p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD),
                     default=None if p.default is inspect.Parameter.empty else p.default, var=p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)) for p in params], ""
    except (Exception, SystemExit) as ex:                                  # noqa: BLE001
        why = f"{type(ex).__name__}: {ex}"
    if python and os.path.realpath(python) != os.path.realpath(sys.executable):
        try:
            r = subprocess.run([python, "-c", _SNIPPET, json.dumps(list(dirs)), entry["suite"], entry["cls"], entry["method"]], capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout.strip().splitlines()[-1]), ""
            why += f"; in {python}: " + (r.stderr.strip().splitlines() or ["exit " + str(r.returncode)])[-1]
        except (OSError, subprocess.SubprocessError, ValueError) as ex:
            why += f"; in {python}: {type(ex).__name__}: {ex}"
    return None, why


def is_ref(v: Any) -> bool:
    """an argument that is the result of another job: {"$result": "<job id>:<path in its result.json>"}, e.g. {"$result": "chain.predict:results"}; it is filled in when the call is made"""
    return isinstance(v, dict) and set(v) == {"$result"}


def refs_in(obj: Any) -> List[str]:
    """every "<job id>:<path>" of a {"$result": ...} anywhere inside the arguments"""
    if is_ref(obj):
        return [obj["$result"]]
    if isinstance(obj, dict):
        return [r for v in obj.values() for r in refs_in(v)]
    if isinstance(obj, (list, tuple)):
        return [r for v in obj for r in refs_in(v)]
    return []


def split_ref(ref: Any) -> Tuple[str, str]:
    """'chain.predict:results' -> ('chain.predict', 'results'); anything that is not 'id:path' -> ('', '')"""
    if isinstance(ref, str) and ":" in ref:
        jid, _, path = ref.partition(":")
        if jid and path:
            return jid, path
    return "", ""


def resolve_refs(obj: Any, jobs_dir: str) -> Any:
    """the arguments with every {"$result": "id:path"} replaced by that value of `<jobs_dir>/<id>/result.json` (a dotted path; list items by number). A missing job, result or
    path is a ValueError that says which"""
    if is_ref(obj):
        jid, path = split_ref(obj["$result"])
        if not jid:
            raise ValueError(f"{obj['$result']!r} is not '<job id>:<path>'")
        f = os.path.join(jobs_dir, jid, "result.json")
        try:
            with open(f, encoding="utf-8") as fh:
                cur = json.load(fh)
        except (OSError, ValueError):
            raise ValueError(f"the result of job {jid} is not there ({f}): the job has not finished, or it returned nothing") from None
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                raise ValueError(f"the result of job {jid} has no {path!r} (stopped at {part!r})")
        return cur
    if isinstance(obj, dict):
        return {k: resolve_refs(v, jobs_dir) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [resolve_refs(v, jobs_dir) for v in obj]
    return obj


def check_args(params: List[Dict[str, Any]], args: Dict[str, Any], skip: Any = ()) -> List[str]:
    """the problems of `args` against the parameters of the method: unknown names, missing required ones, a wrong basic type of a default that is a bool / int / float / list.
    The names in `skip` (arguments that are the result of another job: known only later) count as given and are not type-checked"""
    out = []
    names = {p["name"] for p in params}
    if not any(p["var"] for p in params):
        out += [f"args: {k!r} is not a parameter of the method (parameters: {', '.join(sorted(names))})" for k in args if k not in names]
    out += [f"args: {p['name']!r} is required" for p in params if p["required"] and p["name"] not in args and p["name"] not in skip]
    for p in params:
        if p["name"] not in args or p["name"] in skip or p["default"] is None or p["var"]:
            continue
        v, d = args[p["name"]], p["default"]
        if isinstance(d, bool) and not isinstance(v, bool):
            out.append(f"args: {p['name']!r} must be true or false")
        elif isinstance(d, (int, float)) and not isinstance(d, bool) and (isinstance(v, bool) or not isinstance(v, (int, float))):
            out.append(f"args: {p['name']!r} must be a number")
        elif isinstance(d, (list, tuple)) and not isinstance(v, (list, tuple)):
            out.append(f"args: {p['name']!r} must be a list")
    return out


def fill(template: str, values: Dict[str, Any]) -> Optional[str]:
    """a data reference from its template and the arguments; None when an argument it names is missing or has no such part"""
    try:
        return template.format(**values)
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None


def _walk(path: str, args: Dict[str, Any]) -> Any:
    cur: Any = args
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def id_from(path: str, args: Dict[str, Any]) -> str:
    """'job.id' -> args['job']['id'] (a text, else '')"""
    cur = _walk(path, args)
    return cur if isinstance(cur, str) else ""


def ids_from(path: str, args: Dict[str, Any]) -> List[str]:
    """'job.after' -> args['job']['after'] as a list of texts (one text is a list of one; anything else is empty)"""
    cur = _walk(path, args)
    return [cur] if isinstance(cur, str) and cur else [x for x in cur if isinstance(x, str)] if isinstance(cur, list) else []

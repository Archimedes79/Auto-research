"""Where things are: the repository, a problem, an experiment, a run, and the machine (data and run roots, interpreters, GPU slots).

The machine file (~/.research/machine.json, or the file RESEARCH_MACHINE names) is the only place that knows absolute paths; records hold the
placeholders {data} and {runs} instead, so they read the same on every machine."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(_HERE)                           # autoresearch/
RUN_ID = re.compile(r"^(E\d+)\.([a-z][a-z0-9]*)_s(\d+)$")
EXP_ID = re.compile(r"^E(\d+)$")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ResearchError(Exception):
    """a refusal or a wrong input: the command line prints it and exits 1"""


def repo() -> str:
    """the live checkout: the records are written here. A worker started from a snapshot is told by RESEARCH_REPO"""
    return os.path.abspath(os.environ.get("RESEARCH_REPO") or os.path.dirname(CORE))


def problem_dir(problem: str, root: Optional[str] = None) -> str:
    if not NAME.match(problem or ""):
        raise ResearchError(f"problem name {problem!r}: lower-case letters, digits and _")
    path = os.path.join(root or repo(), "problems", problem)
    if not os.path.isfile(os.path.join(path, "problem.json")):
        raise ResearchError(f"no problem {problem!r}: {path}/problem.json is missing")
    return path


def exp_number(exp_id: str) -> int:
    m = EXP_ID.match(exp_id or "")
    if not m:
        raise ResearchError(f"experiment id {exp_id!r}: E<number>")
    return int(m.group(1))


def experiment_dir(problem: str, exp_id: str, root: Optional[str] = None) -> str:
    """problems/<p>/experiments/E###_slug/ of an id (exactly one folder may carry it)"""
    n = exp_number(exp_id)
    base = os.path.join(problem_dir(problem, root), "experiments")
    hits = [d for d in (os.listdir(base) if os.path.isdir(base) else []) if re.match(rf"^E0*{n}(_|$)", d) and os.path.isdir(os.path.join(base, d))]
    if len(hits) != 1:
        raise ResearchError(f"{exp_id}: {'no' if not hits else 'more than one'} experiment folder in {base} ({', '.join(hits) or '-'})")
    return os.path.join(base, hits[0])


def run_id(exp_id: str, arm: str, seed: int) -> str:
    rid = f"{exp_id}.{arm}_s{int(seed)}"
    if not RUN_ID.match(rid):
        raise ResearchError(f"run id {rid!r}: an arm is lower-case letters and digits, starting with a letter")
    return rid


def parse_run(rid: str) -> Tuple[str, str, int]:
    m = RUN_ID.match(rid or "")
    if not m:
        raise ResearchError(f"run id {rid!r}: E<n>.<arm>_s<seed>")
    return m.group(1), m.group(2), int(m.group(3))


def record_dir(problem: str, rid: str, root: Optional[str] = None) -> str:
    """the run's records in git: experiments/E###_slug/runs/<run>/"""
    return os.path.join(experiment_dir(problem, parse_run(rid)[0], root), "runs", rid)


# ------------------------------------------------------------------------------------------------ the machine
MACHINE_KEYS = {"data": str, "runs": str, "python": dict, "gpu_slots": int}


def machine_file() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json"))


def machine() -> Dict[str, Any]:
    path = machine_file()
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except OSError:
        raise ResearchError(f"no machine file {path}: write it with `research machine init --data <dir> --runs <dir> --python <name>=<interpreter>`") from None
    except ValueError as e:
        raise ResearchError(f"{path}: not json ({e})") from None
    for key, typ in MACHINE_KEYS.items():
        if not isinstance(m.get(key), typ) or (typ is int and isinstance(m.get(key), bool)):
            raise ResearchError(f"{path}: '{key}' must be a {typ.__name__}")
    for key in ("data", "runs"):
        if not os.path.isabs(m[key]):
            raise ResearchError(f"{path}: '{key}' must be an absolute path")
    unknown = set(m) - set(MACHINE_KEYS)
    if unknown:
        raise ResearchError(f"{path}: unknown keys {sorted(unknown)}")
    return m


def state_dir() -> str:
    """GPU slots, holds and grants of this machine: next to the machine file, never inside a checkout"""
    path = os.path.join(os.path.dirname(machine_file()), "state")
    os.makedirs(path, exist_ok=True)
    return path


def runs_root(problem: str) -> str:
    return os.path.join(machine()["runs"], problem)


def run_dir(problem: str, rid: str) -> str:
    return os.path.join(runs_root(problem), rid)


def interpreter(name: str) -> str:
    py = machine()["python"].get(name)
    if not py:
        raise ResearchError(f"the machine file names no interpreter {name!r} (python: {sorted(machine()['python'])})")
    return py


def expand(value: Any, problem: str) -> Any:
    """the placeholders {data} and {runs} of a record, made absolute for this machine (in strings, lists and dicts)"""
    m = machine()
    if isinstance(value, str):
        return value.replace("{data}", m["data"]).replace("{runs}", os.path.join(m["runs"], problem))
    if isinstance(value, list):
        return [expand(v, problem) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, problem) for k, v in value.items()}
    return value


def module_path(code_root: str, problem: str) -> str:
    """PYTHONPATH of a process that runs a problem's code from `code_root` (the checkout or a snapshot): the problem folder, the folders its
    problem.json lists under "pythonpath" (flat imports of older code), autoresearch/ and the root (GtGenerator)"""
    pdir = os.path.join(code_root, "problems", problem)
    try:
        extra = read_json(os.path.join(pdir, "problem.json")).get("pythonpath", [])
    except (OSError, ValueError):
        extra = []
    return os.pathsep.join([pdir] + [os.path.join(pdir, d) for d in extra] + [os.path.join(code_root, "autoresearch"), code_root])


# ------------------------------------------------------------------------------------------------ files
def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """atomic: a reader sees the old file or the new one, never half of it"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def rel(path: str, root: Optional[str] = None) -> str:
    return os.path.relpath(path, root or repo()).replace(os.sep, "/")

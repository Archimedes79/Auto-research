"""What every project's adapter would write the same way, for networks that are run folders (config.json and checkpoints ckpt_*.pt):
name, out, network, resolve, checkpoint, identity. A project's adapter defines only sets, source, subject and task (and METRICS, the metric
names an evaluation uses when none are named); whatever it defines itself wins.

The identity of a network covers its config, its checkpoint and the code of the project (every .py outside experiments/, tests/, temp/ and
literature/): stored predictions are reused only when none of the three changed.

The holdout gate. The holdout splits of a project are the list "holdout" of its problem.json. They are closed; one opens only when all of this
holds: the process is the worker's step of a run of `research test` (RESEARCH_HOLDOUT_RUN = RESEARCH_RUN, set by the worker), the research
core has handed over its reader of the TEST log (`set_holdout_gate`; this package imports nothing of `research`), and that reader, asked
again at every read, says the TEST_LOG.md of RESEARCH_PROBLEM names the run under a decided decision. This file is protected (research/gates.py)."""
from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

REQUIRED = ("sets", "source", "subject", "task")
DEFAULTS = ("name", "out", "network", "resolve", "checkpoint", "identity")
CODE_SKIP = {"experiments", "tests", "temp", "literature", "__pycache__", ".git"}
HOLDOUT_GATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "gates.py")      # the file of the one function taken as the gate
_hash_cache: Dict[Tuple[str, int, int], str] = {}
_holdout_gate: Optional[Callable[[str, str], str]] = None    # none handed over: every holdout split stays closed


def file_hash(path: str) -> str:
    """sha256 of a file, remembered per (path, size, mtime) for this process"""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime_ns))
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def code_hash(project: str) -> str:
    """one hash of the project's code: the relative paths and contents of every .py outside the folders of CODE_SKIP"""
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in CODE_SKIP)
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                h.update(os.path.relpath(path, project).replace(os.sep, "/").encode() + b"\0")
                with open(path, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()[:16]


def holdout_sets(project: str) -> List[str]:
    """the holdout splits of a problem; a problem.json that exists but cannot be read closes every split named test"""
    path = os.path.join(project, "problem.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(json.load(f).get("holdout", []))
    except (OSError, ValueError, AttributeError):
        return ["test"]


def set_holdout_gate(refusal: Callable[[str, str], str]) -> None:
    """the research core hands over its reader of the TEST log before a TEST run reads data: refusal(problem, run) -> '' when the log
    opens the holdout for the run, else why not. Only that function is taken: a plain function whose code is `test_refusal` of the
    file HOLDOUT_GATE_FILE. A caller's own yes is refused, whatever it calls itself (a wrapper and a callable object have other code)"""
    global _holdout_gate
    code = refusal.__code__ if inspect.isfunction(refusal) else None
    if code is None or code.co_name != "test_refusal" or not os.path.isfile(code.co_filename) or not os.path.isfile(HOLDOUT_GATE_FILE) \
            or not os.path.samefile(code.co_filename, HOLDOUT_GATE_FILE):
        raise PermissionError(f"the holdout gate is research.gates.test_refusal of the research core ({HOLDOUT_GATE_FILE}), not {refusal!r}")
    _holdout_gate = refusal


def holdout_open() -> bool:
    """the worker's step of `research test` (RESEARCH_HOLDOUT_RUN = RESEARCH_RUN) for a run that the TEST log of RESEARCH_PROBLEM names under
    a decided decision, as the gate handed over says now; no gate, or whatever cannot be checked, keeps the holdout closed"""
    run = os.environ.get("RESEARCH_HOLDOUT_RUN", "")
    if not run or run != os.environ.get("RESEARCH_RUN", "") or _holdout_gate is None:
        return False
    try:
        return _holdout_gate(os.environ.get("RESEARCH_PROBLEM", ""), run) == ""
    except Exception:                                           # noqa: BLE001 - no problem, no log: closed
        return False


def guard(split: str, project: str) -> None:
    """for a problem's own data reader (training included): refuse a holdout split outside a run of `research test`"""
    if split in holdout_sets(project) and not holdout_open():
        raise PermissionError(f"the split {split!r} is a holdout split: it opens only in a run of `research test`")


class Defaults:
    def __init__(self, project: str):
        self.project = project

    def runs(self) -> str:
        return os.environ.get("RESEARCH_RUNS") or os.path.join(self.project, "runs")

    def name(self, net: str) -> str:
        return os.path.basename(str(net).rstrip("/\\"))

    def out(self, net: str) -> str:
        fresh = os.environ.get("RESEARCH_EVAL_OUT")             # the anchor test: a fresh place, nothing stored is reused
        return os.path.join(fresh, self.name(net)) if fresh else os.path.join(net, "eval")

    def network(self, name: str) -> str:
        return os.path.join(self.runs(), name)

    def resolve(self, text: str) -> str:
        """a run folder given as a path or as a name in the run root; anything else is passed on as it is (the adapter's functions decide)"""
        for p in (text, os.path.join(self.runs(), str(text))):
            if os.path.isfile(os.path.join(p, "config.json")):
                return os.path.abspath(p)
        return text

    def checkpoint(self, net: str) -> str:
        found = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
        if not found:
            raise OSError(f"{net}: no checkpoint ckpt_*.pt")
        return found[-1]

    def identity(self, net: str, checkpoint=None) -> Dict[str, Any]:
        ck = (checkpoint or self.checkpoint)(net)
        return {"config": file_hash(os.path.join(net, "config.json"))[:16], "checkpoint": os.path.basename(ck), "checkpoint_hash": file_hash(ck)[:16],
                "code": code_hash(self.project), "engine": code_hash(os.path.dirname(os.path.abspath(__file__)))}


class Adapter:
    """a project's adapter module with the defaults for what it does not define"""

    def __init__(self, mod: Any, project: str):
        self._mod, self._defaults, self.project = mod, Defaults(project), project

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        if hasattr(self._mod, key):
            return getattr(self._mod, key)
        if key == "identity" and hasattr(self._mod, "checkpoint"):         # the project's checkpoint choice decides the identity too
            return lambda net: self._defaults.identity(net, self._mod.checkpoint)
        if key in DEFAULTS:
            return getattr(self._defaults, key)
        raise AttributeError(key)

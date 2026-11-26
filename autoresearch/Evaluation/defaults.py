"""What every project's adapter would write the same way, for networks that are run folders (config.json and checkpoints ckpt_*.pt):
name, out, network, resolve, checkpoint, identity. A project's adapter defines only sets, source, subject and task (and METRICS, the metric
names an evaluation uses when none are named); whatever it defines itself wins.

The identity of a network covers its config, its checkpoint and the code of the project (every .py outside experiments/, tests/, temp/ and
literature/): stored predictions are reused only when none of the three changed. The holdout splits of a project are the list "holdout" of its
problem.json; they open only in a run of `research test` (the worker sets RESEARCH_HOLDOUT_RUN to the run it evaluates)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

REQUIRED = ("sets", "source", "subject", "task")
DEFAULTS = ("name", "out", "network", "resolve", "checkpoint", "identity")
CODE_SKIP = {"experiments", "tests", "temp", "literature", "__pycache__", ".git"}
_hash_cache: Dict[Tuple[str, int, int], str] = {}


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
    try:
        with open(os.path.join(project, "problem.json"), encoding="utf-8") as f:
            return list(json.load(f).get("holdout", []))
    except (OSError, ValueError):
        return []


def holdout_open() -> bool:
    run = os.environ.get("RESEARCH_HOLDOUT_RUN", "")
    return bool(run) and run == os.environ.get("RESEARCH_RUN", "")


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

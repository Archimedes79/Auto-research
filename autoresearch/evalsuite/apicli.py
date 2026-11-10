"""The command line of the interface (api.py): every method of `Evaluation` is a command with the same name. `python -m evalsuite Evaluation.run runs/t23_C_s1 small.val`,
`Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable`, `Evaluation.compare A B` (b - a); `qual-frame` is `qual_frame`. A method that no older command of this tool uses also works
without the class name (`run`, `check`, `by`, ...; `list`, `quant`, `probe`, `selftest` need `Evaluation.`). A parameter without a default (and `network`, `id`, `results`, `a`, `b`, `frame`,
`frames`) is positional, every other one an option (`--gt-key v8`, `--workers 0`, `--metrics maps,boxes,curves` for a list). `--project` names the project folder (default
AR_PROJECT), `--adapter` its adapter module, `--json` prints exactly what the method returns (nothing else on stdout: progress goes to stderr), otherwise a short text.
`methods` lists the methods. Exit code: 0 ok, 1 `check` found problems, 2 an error (the message goes to stderr; EVALSUITE_TRACEBACK=1 shows the traceback)."""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
from typing import Any, Optional, Sequence

from .api import Evaluation, EvaluationError

LEGACY = ("list", "quant", "qual", "probe", "selftest", "selftest-adapter")                # words of the older command line (cli.py): they need `Evaluation.` here
POSITIONAL = ("network", "id", "results", "a", "b", "frame", "frames")          # the things a command is about: positional even when they have a default (`run NETWORK SET`)


def _clean(obj: Any) -> Any:
    """json-safe: NaN / inf become null, numpy scalars numbers, tuples lists"""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):                 # numpy arrays and scalars
        return _clean(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _text(obj: Any, indent: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{indent}{k}:")
                _text(v, indent + "  ")
            else:
                print(f"{indent}{k}: {v:.6g}" if isinstance(v, float) else f"{indent}{k}: {v}")
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                print(f"{indent}- " + ", ".join(f"{a}={b:.6g}" if isinstance(b, float) else f"{a}={b}" for a, b in x.items()))
            else:
                print(f"{indent}- {x}")
    else:
        print(f"{indent}{obj}")


def _signature(name: str) -> str:
    return ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                     for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items() if k != "self")


def methods_text() -> None:
    for name in Evaluation.METHODS:
        print(f"Evaluation.{name}({_signature(name)})")
        print("    " + (inspect.getdoc(getattr(Evaluation, name)) or "").splitlines()[0])


def _argparser(name: str) -> argparse.ArgumentParser:
    fn = getattr(Evaluation, name)
    ap = argparse.ArgumentParser(prog=f"python -m evalsuite Evaluation.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, p in inspect.signature(fn).parameters.items():
        if k == "self":
            continue
        flag = "--" + k.replace("_", "-")
        if p.default is inspect.Parameter.empty:
            ap.add_argument(k)
        elif k in POSITIONAL:
            ap.add_argument(k, nargs="?", default=p.default, type=type(p.default) if isinstance(p.default, int) else str)
        elif isinstance(p.default, (tuple, list)):
            ap.add_argument(flag, dest=k, default=",".join(p.default), help="comma separated (default %(default)s)")
        elif isinstance(p.default, (int, float)):
            ap.add_argument(flag, dest=k, type=type(p.default), default=p.default)
        else:
            ap.add_argument(flag, dest=k, default=p.default)
    ap.add_argument("--project", default=None, help="project folder (default: AR_PROJECT)")
    ap.add_argument("--adapter", default=None, help="the adapter module (default: <project>/evaluation/evalsuite.json)")
    ap.add_argument("--json", action="store_true", help="print exactly what the method returns")
    return ap


def dispatch(argv: Sequence[str]) -> Optional[int]:
    """the exit code when argv names a method (`Evaluation.run`, or `run` when no older command has the word), else None; `methods` lists them"""
    if not argv:
        return None
    if argv[0] == "methods":
        methods_text()
        return 0
    head = argv[0]
    prefixed = head.startswith("Evaluation.")
    name = (head.split(".", 1)[-1] if prefixed else head).replace("-", "_")
    if name not in Evaluation.METHODS or (not prefixed and head in LEGACY):
        return None
    a = _argparser(name).parse_args(argv[1:])
    kw = {}
    for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items():
        if k != "self":
            v = getattr(a, k)
            kw[k] = tuple(x for x in v.split(",") if x) if isinstance(p.default, (tuple, list)) else v
    try:
        with contextlib.redirect_stdout(sys.stderr):                                     # whatever the engine or the project prints is not the result
            result = _clean(getattr(Evaluation(a.project, a.adapter), name)(**kw))
    except (Exception, SystemExit) as e:                                                 # an error of the data, the network or the machine is a message and exit 2, never a traceback
        if os.environ.get("EVALSUITE_TRACEBACK"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str) else str(e.code if isinstance(e, SystemExit) and e.code is not None else e)
        print(f"error: {type(e).__name__ + ': ' if not isinstance(e, EvaluationError) else ''}{msg}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(result, indent=1, allow_nan=False))
    else:
        _text(result)
    return 1 if name == "check" and not result.get("ok") else 0

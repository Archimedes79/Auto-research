"""The command line: every method of `Evaluation` (api.py, METHODS) is a command of the same name, generated from its signature. From autoresearch/:

    python -m Evaluation run t21_ctl_s0 small.val --gt-key v13 --metrics maps,boxes
    python -m Evaluation show-numbers ID --match drivable --json        (`show_numbers` and `Evaluation.show_numbers` work too)
    python -m Evaluation methods                    the commands with their arguments
    python -m Evaluation selftest [word ...]        the tests (selftest.py)
A parameter without a default, and `network`, `id`, `results`, `a`, `b`, `frame`, `frames`, is positional; every other one is an option (`--gt-key v8`, `--workers 0`); a list is
repeatable or comma separated (`--taps a,b`). `--project DIR` (default AR_PROJECT), `--adapter MODULE` (default: <project>/evaluation/evaluation.json), `--json`: stdout holds exactly
what the method returns (whatever the engine prints goes to stderr; NaN and Infinity become null). Exit 0 done, 1 the result says ok: False (`check`), 2 an error: a message on stderr
(EVALUATION_TRACEBACK=1 shows the traceback)."""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
from typing import Any, List, Optional, Sequence

from .api import Evaluation, EvaluationError

POSITIONAL = ("network", "id", "results", "a", "b", "frame", "frames")          # the things a command is about: positional even when they have a default


def clean(obj: Any, strict: bool = True) -> Any:
    """numpy scalars become numbers, numpy arrays and tuples lists, keys texts; strict (json): NaN and Infinity become null"""
    if isinstance(obj, dict):
        return {str(k): clean(v, strict) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, strict) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return clean(obj.tolist(), strict)
    if strict and isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _num(v: Any) -> Any:
    """a float with six digits; a large one (an epoch) keeps its digits, nan and inf are shown as they are"""
    if not isinstance(v, float):
        return v
    return repr(v) if not math.isfinite(v) or abs(v) >= 1e5 else f"{v:.6g}"


def _text(obj: Any, indent: str = "") -> None:
    """the short text of a result: `key: value` lines, lists as `- item`"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{indent}{k}:")
                _text(v, indent + "  ")
            else:
                print(f"{indent}{k}: {_num(v)}")
    elif isinstance(obj, list):
        for x in obj:
            print(f"{indent}- " + (", ".join(f"{a}={_num(b)}" for a, b in x.items()) if isinstance(x, dict) else f"{x}"))
    else:
        print(f"{indent}{_num(obj)}")


def _params(cls: type, name: str) -> List[inspect.Parameter]:
    return [p for k, p in inspect.signature(getattr(cls, name)).parameters.items() if k != "self"]


def _is_list(p: inspect.Parameter) -> bool:
    return isinstance(p.default, (tuple, list)) or "Sequence[" in str(p.annotation)


def parser(cls: type, name: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m Evaluation {name.replace('_', '-')}", description=inspect.getdoc(getattr(cls, name)), formatter_class=argparse.RawDescriptionHelpFormatter)
    for p in _params(cls, name):
        flag = "--" + p.name.replace("_", "-")
        required = p.default is inspect.Parameter.empty
        kind = type(p.default) if type(p.default) in (int, float) else str          # the type an optional value keeps
        if required:
            ap.add_argument(p.name)
        elif p.name in POSITIONAL:
            ap.add_argument(p.name, nargs="?", default=p.default, type=kind)
        elif isinstance(p.default, bool):
            ap.add_argument(("--no-" + flag[2:]) if p.default else flag, dest=p.name, action="store_false" if p.default else "store_true")
        elif _is_list(p):
            ap.add_argument(flag, dest=p.name, action="append", help="repeatable or comma separated" + (f" (default {','.join(p.default)})" if p.default else ""))
        else:
            ap.add_argument(flag, dest=p.name, type=kind, default=p.default)
    ap.add_argument("--project", help="project folder (default: AR_PROJECT)")
    ap.add_argument("--adapter", help="the adapter module (default: <project>/evaluation/evaluation.json)")
    ap.add_argument("--json", action="store_true", help="print exactly what the method returns (nothing else on stdout)")
    return ap


def methods_text(cls: type) -> None:
    """what `methods` prints: the command, the arguments of the method and its first sentence"""
    for name in cls.METHODS:
        args = ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}" for p in _params(cls, name))
        doc = " ".join((inspect.getdoc(getattr(cls, name)) or "").split())
        print(f"{name.replace('_', '-')}  ({cls.__name__}.{name}({args}))")
        print(f"    {doc.split('. ')[0].rstrip('.')}.")


def call(cls: type, name: str, args: Sequence[str]) -> int:
    """run method `name` of `cls(project, adapter)` with the command line `args`; -> the exit code"""
    a = parser(cls, name).parse_args(list(args))
    kwargs = {}
    for p in _params(cls, name):
        v = getattr(a, p.name)
        if _is_list(p):
            v = p.default if v is None else tuple(x for given in v for x in given.split(",") if x)
        kwargs[p.name] = v
    try:
        with contextlib.redirect_stdout(sys.stderr):                       # whatever the engine or the project prints is not the result
            result = getattr(cls(a.project, a.adapter), name)(**kwargs)
        document = json.dumps(clean(result), indent=1, allow_nan=False, default=str) if a.json else None
    except (Exception, SystemExit) as e:                                   # an error of the data, the machine or a file is a message and exit 2, never a traceback
        if isinstance(e, SystemExit) and e.code in (None, 0):              # a method that ends the process on purpose
            return 0
        if os.environ.get("EVALUATION_TRACEBACK", "") not in ("", "0"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str) else str(e.code if isinstance(e, SystemExit) else e)
        print(f"error: {'' if isinstance(e, EvaluationError) else type(e).__name__ + ': '}{msg or repr(e)}", file=sys.stderr)
        return 2
    if a.json:
        print(document)
    else:
        _text(clean(result, strict=False))
    return 1 if isinstance(result, dict) and result.get("ok") is False else 0


def main(argv: Optional[Sequence[str]] = None, cls: type = Evaluation) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "methods":
        methods_text(cls)
        return 0
    if argv[0] == "selftest":
        from .selftest import main as selftest
        return selftest(argv[1:])
    name = argv[0].removeprefix(cls.__name__ + ".").replace("-", "_")
    if name not in cls.METHODS:
        print(f"Evaluation: unknown command {argv[0]!r}; `python -m Evaluation methods` lists them", file=sys.stderr)
        return 2
    return call(cls, name, argv[1:])

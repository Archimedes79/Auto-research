"""`python -m jobrunner call --suite S --cls C --method M --args-file A.json --out R.json --project P`: the child process of a `process` job.

It builds `C(P)` from module S, calls `M(**args)` - the same call as in Python, with the types of the json (lists, dicts, numbers) - and writes what the method returns to R.json.
Exit code as in every suite: 0 done, 1 the result is a dict with ok=False, 2 an error (a message on stderr; R.json then holds {"error": ...}). What the method prints goes to stderr."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys

AR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(obj):
    """numpy scalars / arrays and tuples become json, NaN / Infinity null"""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return clean(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m jobrunner call")
    for flag in ("--suite", "--cls", "--method", "--args-file", "--out", "--project"):
        ap.add_argument(flag, required=True)
    a = ap.parse_args(argv)
    for d in (os.path.dirname(AR), AR, a.project):                      # sys.path: the project, autoresearch/, the repository root (GtGenerator)
        if d not in sys.path:
            sys.path.insert(0, d)
    result = None
    try:
        with open(a.args_file, encoding="utf-8") as f:
            args = json.load(f)
        inst = getattr(importlib.import_module(a.suite), a.cls)(a.project)
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(inst, a.method)(**args)
        doc = clean(result)
    except (Exception, SystemExit) as ex:                                  # noqa: BLE001
        if isinstance(ex, SystemExit) and ex.code in (None, 0):
            doc, result = None, None
        else:
            msg = str(ex.code if isinstance(ex, SystemExit) else ex) or repr(ex)
            print(f"error: {type(ex).__name__}: {msg}", file=sys.stderr)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({"error": f"{type(ex).__name__}: {msg}"}, f)
            return 2
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, allow_nan=False, default=str)
    return 1 if isinstance(result, dict) and result.get("ok") is False else 0

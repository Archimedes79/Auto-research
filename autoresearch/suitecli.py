"""The one command-line generator of the suites (the convention: autoresearch/README.md, section 'The interface of the suites'): every method of a suite's class is ONE command, the same call
as in Python. The command line of trainsuite is configured at the end of its api.py, evalsuite/apicli.py and the Runner of bev_fusion/runner are the same few lines, on top of this module.

    cli = suitecli.Cli(Training, "trainsuite", TrainingError, "TRAINSUITE_TRACEBACK", legacy=LEGACY, positional=("job", "i", "a", "b"), need_all=("queue_hold",), text={"check": ...})
    rc = cli.dispatch(argv)          # None when argv names no method of the class (the older argparse commands take over), else the exit code

What the generator does with a method `Class.queue_add(job, start=True)`:
- the command is `Class.queue_add`, `queue_add` or `queue-add` (a word of `legacy` is an older command of the tool: only the form with the class name is taken); `Class.nothing` is an error;
- a parameter without a default is positional (except those named in `option_names`, e.g. `words`), so are the names in `positional` even with a default (`[I]`); a bool is a switch
  (`--no-start` when it is on by default); a list or tuple is repeatable or comma separated (`--taps a,b`, `--kill R1 --kill R2`; a comma inside one value splits it); an int or float keeps
  its type; anything else is a text;
- `--project DIR`, `--json` (exactly what the method returns on stdout, NaN / Infinity / numpy become null / numbers / lists; whatever the method prints goes to stderr) and the options of
  `extra` come with every command; `need_all` methods want an item `I` or `--all` (not both);
- exit codes: 0 done, 1 the result is a dict with ok=False (`check`, the dry run `plan`, any other dry run), 2 an error: a message on stderr, no traceback (a non-empty value of the environment variable `traceback_env` shows it). Every Exception and
  every SystemExit of a method is such a message (a SystemExit with code 0 is a normal end); the suite's own error class is printed without its type name;
- the text of a result: `key: value` lines, lists as `- item`, floats with six digits (a large float, an epoch for example, keeps its digits; nan and inf are shown as they are).
`python3 suitecli.py --selftest` checks the generator on a toy class. Standard library only."""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def command(method: str) -> str:
    """the command of a method: queue_add -> queue-add"""
    return method.replace("_", "-")


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


def _num(v: float) -> str:
    return repr(v) if not math.isfinite(v) or abs(v) >= 1e5 else f"{v:.6g}"


def text(obj: Any, indent: str = "") -> None:
    """the short text of a result: `key: value` lines, lists as `- item`, floats with six digits"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{indent}{k}:")
                text(v, indent + "  ")
            else:
                print(f"{indent}{k}: {_num(v) if isinstance(v, float) else v}")
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                print(f"{indent}- " + ", ".join(f"{a}={_num(b) if isinstance(b, float) else b}" for a, b in x.items()))
            else:
                print(f"{indent}- {x}")
    else:
        print(f"{indent}{_num(obj) if isinstance(obj, float) else obj}")


def summary(fn: Callable) -> str:
    """the first sentence of the docstring of a method"""
    doc = " ".join((inspect.getdoc(fn) or "").split())
    return doc.split(". ")[0].rstrip(".") + "." if doc else ""


def _params(fn: Callable) -> List[inspect.Parameter]:
    return [p for k, p in inspect.signature(fn).parameters.items() if k != "self"]


def method_list(cls: type) -> List[Dict[str, str]]:
    """[{name, arguments, summary}] of the methods named in cls.METHODS; arguments as `network, frames='val', start=True`"""
    out = []
    for name in cls.METHODS:
        fn = getattr(cls, name)
        args = ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}" for p in _params(fn))
        out.append(dict(name=name, arguments=args, summary=summary(fn)))
    return out


class Cli:
    def __init__(self, cls: type, prog: str, error: type, traceback_env: str, legacy: Sequence[str] = (), positional: Sequence[str] = (), option_names: Sequence[str] = (),
                 need_all: Sequence[str] = (), text: Optional[Dict[str, Callable]] = None, extra: Sequence[Tuple[str, str]] = (), make: Optional[Callable] = None):
        self.cls, self.prog, self.error, self.traceback_env = cls, prog, error, traceback_env
        self.legacy = {w.replace("-", "_") for w in legacy}
        self.positional, self.option_names, self.need_all = tuple(positional), tuple(option_names), tuple(need_all)
        self.text, self.extra = dict(text or {}), tuple(extra)
        self.make = make or (lambda cls, a: cls(a.project))

    def methods_text(self) -> None:
        """what `methods` prints: the command, the arguments of the method and its first sentence"""
        for m in method_list(self.cls):
            word = f"{self.cls.__name__}.{m['name']}" if m["name"] in self.legacy else command(m["name"])
            print(f"{word}  ({self.cls.__name__}.{m['name']}({m['arguments']}))")
            print(f"    {m['summary']}")

    def parser(self, name: str) -> argparse.ArgumentParser:
        fn = getattr(self.cls, name)
        ap = argparse.ArgumentParser(prog=f"python -m {self.prog} {self.cls.__name__}.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)
        for p in _params(fn):
            flag = "--" + p.name.replace("_", "-")
            required = p.default is inspect.Parameter.empty
            kind = type(p.default) if isinstance(p.default, (int, float)) and not isinstance(p.default, bool) else str       # the type an optional value keeps
            if p.name in self.positional or (required and p.name not in self.option_names):
                if required:
                    ap.add_argument(p.name)
                else:
                    ap.add_argument(p.name, nargs="?", default=p.default, type=kind)
            elif isinstance(p.default, bool):
                if p.default:
                    ap.add_argument("--no-" + p.name.replace("_", "-"), dest=p.name, action="store_false", help=f"switch {p.name} off")
                else:
                    ap.add_argument(flag, dest=p.name, action="store_true")
            elif isinstance(p.default, (tuple, list)):
                ap.add_argument(flag, dest=p.name, action="append", default=None, help="repeatable or comma separated" + (f" (default {','.join(map(str, p.default))})" if p.default else ""))
            else:
                ap.add_argument(flag, dest=p.name, type=kind, required=required, default=None if required else p.default)
        ap.add_argument("--project", default=None, help="project folder (default: AR_PROJECT)")
        for flag, helptext in self.extra:
            ap.add_argument(flag, default=None, help=helptext)
        ap.add_argument("--json", action="store_true", help="print exactly what the method returns (nothing else on stdout)")
        if name in self.need_all:
            ap.add_argument("--all", action="store_true", help="every item (instead of naming one)")
        return ap

    def arguments(self, name: str, ap: argparse.ArgumentParser, a: argparse.Namespace) -> Dict[str, Any]:
        """the keyword arguments of the method from the parsed command line"""
        if name in self.need_all:
            item = getattr(a, "i", None)
            if a.all and item:
                ap.error("give an item or --all, not both")
            if not a.all and not item:
                ap.error("name the item (I) or give --all: without an item the call means every item")
            if a.all:
                a.i = None
        kw = {}
        for p in _params(getattr(self.cls, name)):
            v = getattr(a, p.name)
            if isinstance(p.default, (tuple, list)):
                given = [v] if isinstance(v, str) else v                         # a repeated option, or the text of a positional; not given: the default
                v = p.default if given is None else type(p.default)(x for g in given for x in str(g).split(",") if x)
            kw[p.name] = v
        return kw

    def call(self, name: str, args: Sequence[str]) -> int:
        """run method `name` with the command line `args`; -> the exit code"""
        ap = self.parser(name)
        a = ap.parse_args(list(args))
        kw = self.arguments(name, ap, a)
        try:
            with contextlib.redirect_stdout(sys.stderr):                       # whatever the engine or the project prints is not the result
                result = getattr(self.make(self.cls, a), name)(**kw)
            document = json.dumps(clean(result), indent=1, allow_nan=False, default=str) if a.json else None
        except (Exception, SystemExit) as e:                                  # an error of the data, the machine or a file is a message and exit 2, never a traceback
            if isinstance(e, SystemExit) and e.code in (None, 0):             # a method that ends the process on purpose
                return 0
            if os.environ.get(self.traceback_env, "") not in ("", "0"):
                raise
            msg = e.args[0] if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str) else str(e.code if isinstance(e, SystemExit) else e)
            print(f"error: {'' if isinstance(e, self.error) else type(e).__name__ + ': '}{msg or repr(e)}", file=sys.stderr)
            return 2
        if a.json:
            print(document)
        else:
            self.text.get(name, text)(clean(result, strict=False))
        return 1 if isinstance(result, dict) and result.get("ok") is False else 0

    def dispatch(self, argv: Sequence[str]) -> Optional[int]:
        """-> the exit code when argv names a method (`Class.method`, `method`, `meth-od`; a word of `legacy` only with the class name), else None. `methods` lists them"""
        if not argv:
            return None
        if argv[0] == "methods":
            self.methods_text()
            return 0
        prefixed = argv[0].startswith(self.cls.__name__ + ".")
        name = (argv[0].split(".", 1)[1] if prefixed else argv[0]).replace("-", "_")
        if prefixed and name not in self.cls.METHODS:
            print(f"{self.prog}: unknown command {argv[0]!r}; `python -m {self.prog} methods` lists them", file=sys.stderr)
            return 2
        if name in self.cls.METHODS and (prefixed or name not in self.legacy):
            return self.call(name, argv[1:])
        return None


# ---------------------------------------------------------------------------------------------------------------- self-test
def selftest() -> int:
    from pathlib import PurePosixPath

    class Toy:
        METHODS = ("check", "plan", "add", "hold", "stats", "list", "paths", "pos", "ends")

        def __init__(self, project=None):
            self.project = project

        def check(self, job):
            """look only. More text."""
            return {"ok": job != "bad", "id": job}

        def plan(self, job):
            """a dry run with another name."""
            return {"ok": job != "bad"}

        def add(self, job, start=True, taps=(), seed=1, rate=0.5, note="", dts=()):
            """put a job in. More text."""
            print("progress noise")
            return {"job": job, "start": start, "taps": taps, "seed": seed, "rate": rate, "note": note, "dts": dts, "loss": float("nan"), "when": 1794274693.4974828, "project": self.project}

        def hold(self, words, i=None):
            """hold one or all."""
            return {"words": words, "i": i}

        def stats(self, a, b=None):
            """two ids."""
            raise KeyError("no run 'x'")

        def list(self):
            """an older word."""
            raise SystemExit("no project here")

        def paths(self):
            """a result that is not plain json."""
            return {"p": PurePosixPath("/x")}

        def pos(self, items=(), ratio=0.5, n=None):
            """optional positionals."""
            return {"items": items, "ratio": ratio, "n": n}

        def ends(self):
            """ends the process on purpose."""
            raise SystemExit(0)

    def run(cli, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = cli.dispatch(list(argv))
            except SystemExit as e:
                rc = ("exit", e.code)
        return rc, out.getvalue(), err.getvalue()

    class Err(Exception):
        pass
    cli = Cli(Toy, "toy", Err, "TOY_TRACEBACK", legacy=("list", "old-word"), positional=("job", "i", "a", "b", "items", "ratio"), option_names=("words",), need_all=("hold",))
    rc, out, err = run(cli, "Toy.add", "j1", "--no-start", "--taps", "a,b", "--taps", "c", "--seed", "3", "--rate", "0.25", "--note", "x y", "--json", "--project", "P")
    got = json.loads(out)
    assert rc == 0 and got["start"] is False and got["taps"] == ["a", "b", "c"] and got["seed"] == 3 and got["rate"] == 0.25 and got["loss"] is None and got["project"] == "P", (rc, out)
    assert "progress noise" in err and "progress" not in out, "the method's own prints go to stderr"
    rc, out, err = run(cli, "add", "j1")
    assert rc == 0 and "start: True" in out and "taps: []" in out and "loss: nan" in out and "when: 1794274693.4974828" in out and "rate: 0.5" in out, out
    assert run(cli, "add", "j1", "--dts", "")[0] == 0 and json.loads(run(cli, "add", "j1", "--dts", "", "--json")[1])["dts"] == [], "an empty list can be passed"
    assert run(cli, "queue-add")[0] is None and run(cli, "nothing")[0] is None and run(cli, "")[0] is None
    rc, out, err = run(cli, "Toy.nothing")
    assert rc == 2 and "unknown command 'Toy.nothing'" in err and run(cli, "Toy.")[0] == 2, err
    assert run(cli, "check", "ok1")[0] == 0 and run(cli, "check", "bad")[0] == 1, "check exits 1 when ok is False"
    assert run(cli, "plan", "ok1")[0] == 0 and run(cli, "plan", "bad")[0] == 1, "so does every other method whose result says ok=False"
    assert run(cli, "add", "j1")[0] == 0, "a result without ok is done"
    assert run(cli, "list")[0] is None, "an older word needs the class name"
    rc, out, err = run(cli, "Toy.list")
    assert rc == 2 and err == "error: SystemExit: no project here\n", err
    assert run(cli, "ends")[0] == 0, "a SystemExit with code 0 is a normal end"
    rc, out, err = run(cli, "stats", "x")
    assert rc == 2 and err == "error: KeyError: no run 'x'\n", err
    assert run(cli, "hold", "--words", "W")[0] == ("exit", 2), "no item and no --all is a usage error"
    assert run(cli, "hold", "", "--words", "W")[0] == ("exit", 2), "an empty item is no item"
    assert run(cli, "hold", "7", "--all", "--words", "W")[0] == ("exit", 2), "an item and --all"
    rc, out, err = run(cli, "hold", "7", "--words", "W", "--json")
    assert rc == 0 and json.loads(out) == {"words": "W", "i": "7"}
    rc, out, err = run(cli, "hold", "--all", "--words", "W", "--json")
    assert rc == 0 and json.loads(out)["i"] is None
    assert run(cli, "hold", "--json")[0] == ("exit", 2), "words is required"
    rc, out, err = run(cli, "pos", "ab,cd", "0.25", "--n", "4", "--json")
    assert json.loads(out)["items"] == ["ab", "cd"], out
    assert json.loads(out)["ratio"] == 0.25 and isinstance(json.loads(out)["ratio"], float), "a positional float keeps its type"
    rc, out, err = run(cli, "paths", "--json")
    assert rc == 0 and json.loads(out) == {"p": "/x"}, (rc, out, err)
    rc, out, err = run(cli, "methods")
    assert rc == 0 and "add  (Toy.add(job, start=True, taps=(), seed=1, rate=0.5, note='', dts=()))" in out and "Toy.list  (Toy.list())" in out and "    look only." in out, out
    os.environ["TOY_TRACEBACK"] = "1"
    try:
        try:
            run(cli, "stats", "x")
            raise AssertionError("the traceback variable must re-raise")
        except KeyError:
            pass
    finally:
        del os.environ["TOY_TRACEBACK"]
    assert clean({"a": float("inf"), "b": (1, 2), "c": {3: [float("nan")]}}) == {"a": None, "b": [1, 2], "c": {"3": [None]}}
    assert math.isnan(clean({"a": float("nan")}, strict=False)["a"])
    print("suitecli selftest passed")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)

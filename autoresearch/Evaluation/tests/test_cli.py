"""Known answers of the command line generator (cli.py) on a toy class, and the README's example run as it stands."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from pathlib import PurePosixPath
from typing import Optional, Sequence

from .. import cli
from ..api import EvaluationError
from . import check, raises


class Toy:
    METHODS = ("check", "add", "stats", "said", "paths", "ends")

    def __init__(self, project=None, adapter=None):
        self.project, self.adapter = project, adapter

    def check(self, job):
        """a dry run. More text."""
        return {"ok": job != "bad", "id": job}

    def add(self, network, frames: str = "val", taps: Sequence[str] = (), metrics: Optional[Sequence[str]] = None, workers: int = 0, rate: float = 0.5, force: bool = False,
            keep: bool = True, note: Optional[str] = None):
        """every kind of parameter. More text."""
        print("progress noise")
        return {"network": network, "frames": frames, "taps": taps, "metrics": metrics, "workers": workers, "rate": rate, "force": force, "keep": keep, "note": note,
                "loss": float("nan"), "when": 1794274693.4974828, "project": self.project, "adapter": self.adapter}

    def stats(self, a, b=None):
        """two ids."""
        raise KeyError("no run 'x'")

    def said(self):
        """an error of the evaluation, or of a library that ends the process."""
        raise EvaluationError("say what to fix") if self.project else SystemExit("no project here")

    def paths(self):
        """a result that is not plain json."""
        return {"p": PurePosixPath("/x")}

    def ends(self):
        """ends the process on purpose."""
        raise SystemExit(0)


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv), Toy)
    return code, out.getvalue(), err.getvalue()


@check("command line: positionals and options keep their types, lists are repeatable or comma separated, switches; --json is exactly the result; exit 0 / 1 / 2 with a message")
def _():
    code, out, err = run("Toy.add", "n1", "small.val", "--taps", "a,b", "--taps", "c", "--metrics", "maps,boxes", "--workers", "3", "--rate", "0.25", "--force", "--no-keep",
                         "--json", "--project", "P", "--adapter", "m")
    got = json.loads(out)
    assert code == 0 and got == {"network": "n1", "frames": "small.val", "taps": ["a", "b", "c"], "metrics": ["maps", "boxes"], "workers": 3, "rate": 0.25, "force": True, "keep": False,
                                 "note": None, "loss": None, "when": 1794274693.4974828, "project": "P", "adapter": "m"}, (code, got)
    assert "progress noise" in err and "progress" not in out, "what the method prints goes to stderr"
    code, out, _ = run("add", "n1")
    assert code == 0 and all(line in out for line in ("frames: val", "taps: []", "metrics: None", "force: False", "keep: True", "loss: nan", "when: 1794274693.4974828", "rate: 0.5")), out
    assert run("check", "good")[0] == 0 and run("check", "bad")[0] == 1, "a result that says ok: False is exit 1"
    assert run("stats", "x") == (2, "", "error: KeyError: no run 'x'\n")
    assert run("said") == (2, "", "error: SystemExit: no project here\n") and run("said", "--project", "P") == (2, "", "error: say what to fix\n")
    assert run("ends")[0] == 0 and json.loads(run("paths", "--json")[1]) == {"p": "/x"}
    code, _, err = run("nothing")
    assert code == 2 and "unknown command 'nothing'" in err and run("Toy.nothing")[0] == 2
    code, out, _ = run("methods")
    assert code == 0 and "stats  (Toy.stats(a, b=None))" in out and "    a dry run." in out, out
    os.environ["EVALUATION_TRACEBACK"] = "1"
    try:
        raises(KeyError, "no run", run, "stats", "x")
    finally:
        del os.environ["EVALUATION_TRACEBACK"]
    assert cli.clean({"a": float("inf"), "b": (1, 2), "c": {3: [float("nan")]}}) == {"a": None, "b": [1, 2], "c": {"3": [None]}}
    assert math.isnan(cli.clean({"a": float("nan")}, strict=False)["a"])


@check("README: its example runs as written, and the command line reads what it stored")
def _():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
    with open(path, encoding="utf-8") as f:
        blocks = re.findall(r"```python\n(.*?)```", f.read(), re.S)
    assert len(blocks) == 1, "the README holds one python example"
    saved = {k: os.environ.pop(k) for k in ("RESEARCH_RUNS", "RESEARCH_EVAL_OUT") if k in os.environ}          # the example's runs and results are in its own project
    try:
        names: dict = {}
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            exec(compile(blocks[0], path, "exec"), names)              # noqa: S102
        assert os.path.isfile(names["picture"]) and names["difference"] < 0 and "thr05@e" in out.getvalue(), (names.get("difference"), out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            assert cli.main(["show-numbers", names["a"], "--project", names["project"], "--adapter", "Evaluation.minimal", "--json"]) == 0
        assert json.loads(printed.getvalue())["numbers"]["iou.iou"] > 0.5, "the same id from the command line, in a new object"
    finally:
        os.environ.update(saved)

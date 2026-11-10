"""Layer 1 of runsuite, the command line: nothing but the generator of the suites (suitecli.py) pointed at the classes. Every method of `Runner` (api.py) is a command, and so is every
method of `RunJobs` (jobs.py). No logic here: a new method of a class is a new command without a line in this file.

    python -m runsuite Runner.predict runs/t23_C_s1 small.val --taps fused       the network on frames, no ground truth, stored
    python -m runsuite RunJobs.run job.json                                       a job of the project's runner (run.py: ledger row, GPU slot)
    python -m runsuite methods                                                    all of them

`dispatch(argv)` takes `Class.method` or the short `method` (a method of Runner before one of RunJobs; the names do not overlap) and returns the exit code or None when it knows no such command."""
from __future__ import annotations

from typing import Optional, Sequence

import suitecli

from .api import Runner, RunnerError
from .jobs import CLI as JOBS

RUNNER = suitecli.Cli(Runner, "runsuite", RunnerError, "RUNSUITE_TRACEBACK", positional=("network", "frames", "results", "frame", "name"))
CLIS = (RUNNER, JOBS)                                           # one per class, in the order a short command is looked up


def dispatch(argv: Sequence[str]) -> Optional[int]:
    """-> the exit code of the command, None when no class has it"""
    argv = list(argv)
    if argv and argv[0] == "methods":
        for cli in CLIS:
            cli.methods_text()
        return 0
    for cli in CLIS:
        if argv and argv[0].startswith(cli.cls.__name__ + "."):       # the class is named: that one decides, an unknown method is its error
            return cli.dispatch(argv)
    for cli in CLIS:
        rc = cli.dispatch(argv)
        if rc is not None:
            return rc
    return None

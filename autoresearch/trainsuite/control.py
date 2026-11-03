"""Stopping a run from outside: a CONTROL file in the run directory, read by the trainer at a step boundary.

    suite:    request_stop(run_dir, reason)      writes CONTROL = {"cmd": "stop", "reason": ..., "t": ...}
    trainer:  clear_stale(run_dir)               at its START moves CONTROL and STOPPED of an earlier stop aside (to CONTROL.old, STOPPED.old)
              stop_requested(run_dir)            at every step boundary: the reason (a string, maybe empty) or None
              mark_stopped(run_dir, step, why)   after saving a checkpoint; the trainer then exits with EXIT_STOPPED

The protocol is implemented once, in the training engine (`autoresearch/trainkit/control.py`, standard library only, so a run snapshot carries it); this module re-exports
it and adds `honours_stop`. The exit code is not 0 on purpose: `.done` ("finished correctly") is written only for exit code 0 and a stopped run must not get it.
`honours_stop` looks into the run's code snapshot: a trainer older than this contract ignores CONTROL. A CONTROL written before the trainer's `train()` starts (imports
take 10-40 s) is moved aside as stale.
"""
import os

from trainkit.control import CONTROL, EXIT_STOPPED, STOPPED, clear_stale, mark_stopped, request_stop, stop_requested  # noqa: F401 - the suite's names for the protocol


def honours_stop(run_dir):
    """True / False: the run's code snapshot (`<run>/code/train.py`) has / has not the stop check; None: no snapshot to look at"""
    try:
        with open(os.path.join(run_dir, "code", "train.py"), encoding="utf-8", errors="replace") as f:
            return "_stop_requested" in f.read()
    except OSError:
        return None

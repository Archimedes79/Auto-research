"""The stop protocol between a trainer and the Training: a CONTROL file in the run directory, read by the trainer at a step boundary. Standard library only.

    suite:    request_stop(run_dir, reason)      writes CONTROL = {"cmd": "stop", "reason": ..., "t": ...}
    trainer:  clear_stale(run_dir)               at its START moves CONTROL and STOPPED of an earlier stop aside (to CONTROL.old, STOPPED.old):
                                                 a new process never obeys an old request; no clocks are compared
              stop_requested(run_dir)            at every step boundary: the reason (a string, maybe empty) or None
              mark_stopped(run_dir, step, why)   after saving a checkpoint; the trainer then exits with EXIT_STOPPED

The exit code is not 0 on purpose: `.done` ("finished correctly") is written only for exit code 0 and a stopped run must not get it. This module is the one
place that implements the contract; `autoresearch/Training/control.py` re-exports it. A CONTROL written before the trainer's start is moved aside as stale.
"""
import json
import os
import time

CONTROL, STOPPED, EXIT_STOPPED = "CONTROL", "STOPPED", 75      # 75 = EX_TEMPFAIL


def request_stop(run_dir, reason=""):
    path = os.path.join(run_dir, CONTROL)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cmd": "stop", "reason": reason, "t": time.time()}, f)
    os.replace(tmp, path)
    return path


def stop_requested(run_dir):
    """reason (a string, maybe empty but never None) when CONTROL says stop, else None"""
    try:
        with open(os.path.join(run_dir, CONTROL), encoding="utf-8") as f:
            d = json.load(f)
        return str(d.get("reason", "")) if d.get("cmd") == "stop" else None
    except (OSError, ValueError, AttributeError):
        return None


def clear_stale(run_dir):
    """move the files of an earlier stop aside (nothing is deleted); returns the names moved"""
    moved = []
    for n in (CONTROL, STOPPED):
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            os.replace(p, p + ".old")
            moved.append(n)
    return moved


def mark_stopped(run_dir, step, reason=""):
    with open(os.path.join(run_dir, STOPPED), "w", encoding="utf-8") as f:
        json.dump({"step": step, "reason": reason, "t": time.time()}, f)

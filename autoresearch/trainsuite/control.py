"""Stopping a run from outside: a CONTROL file in the run directory, read by the trainer at a step boundary.

    suite:    request_stop(run_dir, reason)      writes CONTROL = {"cmd": "stop", "reason": ..., "t": ...}
    trainer:  stop_requested(run_dir, since)     the same ten lines are inlined in the trainer (no import: a run trains from a pinned code
                                                 snapshot that does not contain this package); on True it saves a checkpoint, writes
                                                 STOPPED = {"step", "reason", "t"} and exits with EXIT_STOPPED
A CONTROL file counts only when it is NEWER than the start of the trainer process (`since`), so an old stop request never stops a resume.
The exit code is not 0 on purpose: `.done` ("finished correctly") is written only for exit code 0 and a stopped run must not get it.
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


def stop_requested(run_dir, since):
    """reason (a string, maybe empty but never None) when a stop was requested after `since` (time.time() of the process start), else None"""
    path = os.path.join(run_dir, CONTROL)
    try:
        if os.path.getmtime(path) <= since:
            return None
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return str(d.get("reason", "")) if d.get("cmd") == "stop" else None
    except (OSError, ValueError, AttributeError):
        return None


def mark_stopped(run_dir, step, reason=""):
    with open(os.path.join(run_dir, STOPPED), "w", encoding="utf-8") as f:
        json.dump({"step": step, "reason": reason, "t": time.time()}, f)

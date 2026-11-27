"""A wait before a GPU evaluation step: do not start a dump / score next to two trainings (D100). Best effort, no numeric effect.

On 2026-11-14 the pair-head dump of E341.p_s1 ran next to two trainings (GPU 11.6 of 12.3 GB, host load 12) and `E317.g_s0` failed at step 2929 with `CUDA error: unknown error`
(resumed from its checkpoint by the old retry wrapper). The launcher's memory limit (< 11 GB) gates the start of a TRAINING only; the evaluation steps of the chains (dumps) start whenever their chain gets there. `wait_for_room()` is called at the start of that command when the process would use the GPU: it waits while the
GPU memory in use (device wide, every process, read by `nvidia-smi` as the launcher's own gate does: no CUDA context is created by the waiting process) is NOT below `max_used_mb`, polls
every `poll_s` seconds and goes on after `timeout_s` (a chain never dead-locks; the line it prints says so). An unreadable `nvidia-smi` is "go on, loudly" (an evaluation step is not
held up by a broken tool). nvidia-smi reads the physical GPU 0 whatever CUDA_VISIBLE_DEVICES says (one GPU on this box; with an empty mask the call sites are never reached).
Limits, stated: a dump started at 8.9 GB can still reach 12 GB (the footprint of a dump or a score is not measured), so the gate protects only when a gap of about one training exists;
several waiters that see the same drop start together (the dumps are serialised by the chains' flock, the maps scores are not). Waiting lines go to stdout AND to `<runs>/lane_center/_scratch/gpu_gate.status`
(a launcher that shows the output of a step only at its end would make a waiting chain look hung).

    wait_for_room(max_used_mb=9000, poll_s=60, timeout_s=4 * 3600, who="")  -> seconds waited (about 0 when there is room, no GPU reading, or the gate is off)

Environment: `GPU_GATE_MB` overrides `max_used_mb` (0 = off; a value that is not a number is ignored), `GPU_GATE_TIMEOUT_S` the timeout.
Self-test (CPU only, no GPU is touched): python evaluation/gpu_gate.py selftest (its hand mutants, evaluation/tests/test_gpu_gate.py, are in the tag v1)
"""
import os
import subprocess
import sys
import time
import bflc_paths as BP  # noqa: E402  (where code, data and runs are)

DEFAULT_MB = 9000
STATUS = os.path.normpath(os.path.join(BP.SCRATCH, "gpu_gate.status"))


def used_mb():
    """GPU memory in use in MiB (all processes, the largest value of the GPUs nvidia-smi lists), or None when nvidia-smi cannot be read"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return None


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _say(msg, log, status):
    log(msg, flush=True)
    if status:
        try:
            with open(status, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
        except OSError:
            pass


def wait_for_room(max_used_mb=DEFAULT_MB, poll_s=60.0, timeout_s=4 * 3600.0, who="", _used=None, _sleep=time.sleep, _now=time.monotonic, log=print, status=STATUS):
    """wait while the GPU memory in use is not below max_used_mb; -> seconds waited. `_used`, `_sleep`, `_now`, `log`, `status` are for the tests"""
    max_used_mb = _env_float("GPU_GATE_MB", max_used_mb)
    timeout_s = _env_float("GPU_GATE_TIMEOUT_S", timeout_s)
    if max_used_mb <= 0:
        return 0.0
    used = _used or used_mb
    t0 = _now()
    n = 0
    while True:
        u = used()
        if u is None:
            _say(f"[gpu_gate] {who}: nvidia-smi unreadable: going on", log, status)
            return _now() - t0
        if u < max_used_mb:
            if n:
                _say(f"[gpu_gate] {who}: room after {_now() - t0:.0f} s ({u:.0f} MiB in use)", log, status)
            return _now() - t0
        if _now() - t0 >= timeout_s:
            _say(f"[gpu_gate] {who}: {u:.0f} MiB still in use after {timeout_s:.0f} s (limit {max_used_mb:.0f}): going on", log, status)
            return _now() - t0
        if n % 10 == 0:
            _say(f"[gpu_gate] {who}: {u:.0f} MiB in use (limit {max_used_mb:.0f}): waiting", log, status)
        n += 1
        _sleep(poll_s)


def _selftest():
    out = []
    lg = lambda s, **k: out.append(s)  # noqa: E731
    now = [0.0]
    sleep = lambda s: now.__setitem__(0, now[0] + s)  # noqa: E731
    clock = lambda: now[0]  # noqa: E731
    kw = dict(_sleep=sleep, _now=clock, log=lg, status=None)
    os.environ.pop("GPU_GATE_MB", None)
    os.environ.pop("GPU_GATE_TIMEOUT_S", None)
    assert wait_for_room(_used=lambda: 5000.0, **kw) == 0.0 and not out                                    # room at once: no wait, no log
    assert wait_for_room(_used=lambda: 9000.0, timeout_s=0, **kw) == 0.0 and any("going on" in s for s in out)   # the limit itself is NOT room (the launcher's `<`)
    out.clear()
    assert wait_for_room(_used=lambda: None, **kw) == 0.0 and any("unreadable" in s for s in out)           # unreadable: go on, loudly
    out.clear()
    seq = iter([11600.0, 11600.0, 11000.0, 8000.0])                                                          # falls after three polls: waits 180 s and says so
    w = wait_for_room(_used=lambda: next(seq), poll_s=60, **kw)
    assert abs(w - 180.0) < 1e-6 and any("waiting" in s for s in out) and any("room after" in s for s in out), (w, out)
    now[0] = 0.0
    out.clear()
    w = wait_for_room(_used=lambda: 11600.0, poll_s=60, timeout_s=600, **kw)                                 # never falls: goes on after the timeout and says so
    assert abs(w - 600.0) < 1e-6 and any("going on" in s for s in out) and sum("waiting" in s for s in out) == 1, (w, out)
    os.environ["GPU_GATE_MB"] = "0"                                                                          # the environment switches it off
    assert wait_for_room(_used=lambda: 99999.0, **kw) == 0.0
    os.environ["GPU_GATE_MB"] = "12000"                                                                      # ... or overrides the limit
    assert wait_for_room(_used=lambda: 11600.0, **kw) == 0.0
    os.environ["GPU_GATE_MB"] = "not a number"                                                               # ... and a bad value is ignored (the default limit stays)
    now[0] = 0.0
    assert abs(wait_for_room(_used=lambda: 11600.0, poll_s=60, timeout_s=120, **kw) - 120.0) < 1e-6
    os.environ.pop("GPU_GATE_MB")
    print("gpu_gate selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print(__doc__)

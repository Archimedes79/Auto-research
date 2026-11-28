"""jtiming.py - the host / GPU timing protocol of `jtoken_eval.py timing`: a forward timed by two CUDA events (or `perf_counter` on the CPU), the host load
and the GPU state before and after, and the rule that calls a timing QUIET. The functions are those of v1's junction timer (E342 / E351, in the tag v1 with
its marker, kernel and CUDA-graph commands, which need the junction system that did not move).

    import jtiming as JTG
    out, ms = JTG.timed(lambda: model(batch), cuda)
    quiet, line = JTG.quiet_verdict(JTG.load1(), load_after, JTG.gpu_state(), gpu_after)     # QUIET: both loads below QUIET_LOAD and the GPU idle before and after

Test: `python jtiming.py` (selftest, CPU).
"""
import os
import subprocess
import time

import numpy as np
import torch

QUIET_LOAD = 3.0                                                                                           # E351: the 1-minute load average (12 cores) below which the host is quiet


def timed(fn, cuda):
    """-> (fn(), ms): two CUDA events with a sync in front and behind (the protocol of this module), or `perf_counter` on the CPU (moved here from `jrecur_time.py`, archive tag archive/jtok2-jrecur-2026-11-18)"""
    if not cuda:
        t = time.perf_counter()
        r = fn()
        return r, (time.perf_counter() - t) * 1000.0
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    r = fn()
    e1.record()
    torch.cuda.synchronize()
    return r, e0.elapsed_time(e1)


def load1():
    """the 1-minute load average of the host (/proc/loadavg), None when it cannot be read"""
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def gpu_idle(state):
    """the GPU counts as idle when `gpu_state()` could read it and no sample showed more than 5 % utilisation"""
    return state is not None and "error" not in state and state["util_max"] <= 5.0


def quiet_verdict(load_before, load_after, gpu_before, gpu_after, limit=QUIET_LOAD):
    """-> (quiet, line): QUIET when both load averages are below `limit` AND the GPU was idle before and after (`gpu_idle`), else NOT QUIET (an unreadable value is not quiet; a run forced onto a busy GPU is never quiet): the speed half of E351
    is then INCONCLUSIVE"""
    quiet = load_before is not None and load_after is not None and load_before < limit and load_after < limit and gpu_idle(gpu_before) and gpu_idle(gpu_after)
    txt = lambda s: "unreadable" if s is None or "error" in s else f"util max {s['util_max']:.0f} %"  # noqa: E731
    return quiet, f"host load (1 min) before {load_before} after {load_after}, limit {limit}; GPU before {txt(gpu_before)}, after {txt(gpu_after)} (idle: at most 5 %): {'QUIET' if quiet else 'NOT QUIET'}"


def ancestors():
    """the pids of this process and its parents (a timing started by a research worker has the worker among them)"""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def gpu_others(apps, own):
    """the compute processes on the GPU that are not ours: apps = the lines `pid, MiB` of `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` (MiB may be [N/A]),
    own = the pids of this process and its parents -> [pid]"""
    pids = [int(ln.split(",")[0]) for ln in apps.splitlines() if ln.split(",")[0].strip().isdigit()]
    return [p for p in pids if p not in own]


def gpu_free(limit_mib=2000):
    """(ok, why): the rule of a timing by hand: nvidia-smi shows less than `limit_mib` used and no compute process on the GPU but this process and its parents (a step of a research worker that runs a
    network is such a process, whoever started it). The utilisation before and after the timing is `quiet_verdict`'s (the after-sample waits 2.5 s for our own kernels)"""
    try:
        smi = lambda q: subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout  # noqa: E731
        used = float(smi("--query-gpu=memory.used").strip().split("\n")[0])
        others = gpu_others(smi("--query-compute-apps=pid,used_memory"), ancestors())
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read the GPU: {e}"
    if used >= limit_mib or others:
        return False, f"GPU memory used {used:.0f} MiB (limit {limit_mib}), other compute processes on the GPU {others}"
    return True, f"GPU memory used {used:.0f} MiB, no other compute process on the GPU"


def gpu_state():
    """utilisation (%) and memory used (MiB) of the GPU by nvidia-smi, 5 samples one second apart; None when nvidia-smi is not there"""
    try:
        out = []
        for _ in range(5):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            u, m, t, c = [float(x) for x in r.stdout.strip().split("\n")[0].split(",")]
            out.append((u, m, t, c))
            time.sleep(1.0)
        a = np.array(out)
        return dict(util_max=float(a[:, 0].max()), mem_mib=float(a[:, 1].mean()), temp_c=float(a[:, 2].mean()), sm_clock_mhz=float(a[:, 3].mean()))
    except Exception as e:                                                                                    # noqa: BLE001
        return dict(error=str(e))


def selftest():
    # the quiet rule, known answers (the host is quiet below a load of 3 before AND after, the GPU idle before and after)
    idle, busy, broken = dict(util_max=0.0, mem_mib=1000.0), dict(util_max=60.0, mem_mib=9000.0), dict(error="no nvidia-smi")
    assert quiet_verdict(2.9, 2.9, idle, idle)[0] and not quiet_verdict(2.9, 3.0, idle, idle)[0] and not quiet_verdict(3.5, 1.0, idle, idle)[0] and not quiet_verdict(None, 1.0, idle, idle)[0]
    assert quiet_verdict(1.0, 1.0, idle, idle)[1].endswith(": QUIET") and quiet_verdict(3.0, 1.0, idle, idle)[1].endswith(": NOT QUIET")
    for g in ((busy, idle), (idle, busy), (idle, broken), (None, idle)):                                      # the GPU counts as much as the load: a busy, unreadable or missing state is not quiet
        assert not quiet_verdict(1.0, 1.0, *g)[0], g
    assert gpu_idle(dict(util_max=5.0)) and not gpu_idle(dict(util_max=5.1)) and not gpu_idle(broken) and not gpu_idle(None)
    assert "util max 60 %" in quiet_verdict(1.0, 1.0, busy, idle)[1]
    # the CPU path of `timed`: the result of fn and its wall time
    r, ms = timed(lambda: time.sleep(0.02) or 7, False)
    assert r == 7 and 15.0 < ms < 1000.0, (r, ms)
    print("jtiming selftest ok")


if __name__ == "__main__":
    selftest()

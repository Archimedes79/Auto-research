"""Scratch: run ONE command only inside Mirko's away windows and only when the GPU and the RAM are free.

Windows (Mirko, 2026-11-05: "Night and work"): every night 22:00 - 07:00, and Monday - Friday 08:30 - 17:30.
A job starts only if it is expected to end before its window closes (--hours + 10 min margin), the GPU shows less than --gpu-max-mb used and less than
--gpu-max-util % busy and NO CUDA compute process (nvidia-smi --query-compute-apps; another chat's job, however small) on two polls in a row (another
chat's job is never touched or waited out by force: the gate just waits), and MemAvailable is at least --min-ram-gb. If the window closes while the job runs, --stop-file gets the JSON {"cmd": "stop", ...} 5 minutes before the end (trainkit's CONTROL
contract: checkpoint, STOPPED, exit 75; a plain word is ignored); nothing is killed.

    python3 temp/t22/window_gate.py --hours 0.5 [--stop-file runs/NAME/CONTROL] -- <command> [args]
    python3 temp/t22/window_gate.py --selftest
Run it with nohup inside WSL so that it survives the shell (WSL ends with an idle session otherwise). Scratch: promote with a test only if a card names it."""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time

NIGHT = (dt.time(22, 0), dt.time(7, 0))
WORK = (dt.time(8, 30), dt.time(17, 30))


def window_end(now):
    """end of the away window that contains `now`, or None when `now` is outside every window"""
    t, wd = now.time(), now.weekday()                      # Monday = 0
    if t >= NIGHT[0]:
        return dt.datetime.combine(now.date() + dt.timedelta(days=1), NIGHT[1])
    if t < NIGHT[1]:
        return dt.datetime.combine(now.date(), NIGHT[1])
    if wd < 5 and WORK[0] <= t < WORK[1]:
        return dt.datetime.combine(now.date(), WORK[1])
    return None


def fits(now, hours, margin_min=10):
    end = window_end(now)
    return end is not None and now + dt.timedelta(hours=hours, minutes=margin_min) <= end


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    mem, util = (float(x) for x in out.strip().splitlines()[0].split(","))
    return mem, util


def cuda_procs():
    """number of CUDA compute processes on the GPU (any chat's job, however small; the desktop's graphics clients are not listed here)"""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return len([x for x in out.splitlines() if x.strip()])


def ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def log(msg):
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def selftest():
    cases = [("2026-11-05 00:24", 1.0, True), ("2026-11-05 06:00", 1.0, False), ("2026-11-05 06:00", 0.5, True), ("2026-11-05 07:30", 0.1, False),
             ("2026-11-05 08:45", 2.5, True), ("2026-11-05 15:30", 2.0, False), ("2026-11-05 22:00", 4.0, True), ("2026-11-06 23:30", 6.0, True),
             ("2026-11-07 10:00", 0.1, False), ("2026-11-08 02:00", 3.0, True), ("2026-11-09 08:29", 0.1, False), ("2026-11-09 08:30", 0.1, True)]
    bad = 0
    for ts, h, want in cases:
        got = fits(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M"), h)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), ts, f"{h} h", "fits" if got else "no", f"(window end {window_end(dt.datetime.strptime(ts, '%Y-%m-%d %H:%M'))})")
    print("selftest", "passed" if not bad else f"FAILED {bad}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0, help="expected duration of the job")
    ap.add_argument("--gpu-max-mb", type=float, default=2500.0)
    ap.add_argument("--gpu-max-util", type=float, default=20.0)
    ap.add_argument("--min-ram-gb", type=float, default=8.0)
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    calm = 0
    while True:
        now = dt.datetime.now()
        if fits(now, a.hours):
            mem, util = gpu_state()
            procs = cuda_procs()
            ok = mem < a.gpu_max_mb and util < a.gpu_max_util and ram_gb() >= a.min_ram_gb and procs == 0
            calm = calm + 1 if ok else 0
            log(f"in window until {window_end(now):%H:%M}; GPU {mem:.0f} MB {util:.0f} %, {procs} CUDA process(es), RAM {ram_gb():.1f} GB: {'free' if ok else 'busy'} ({calm}/2)")
            if calm >= 2:
                break
        else:
            calm = 0
            log("outside the windows or not enough time left: waiting")
        time.sleep(a.poll_s)
    end = window_end(dt.datetime.now())
    log("starting: " + " ".join(cmd))
    p = subprocess.Popen(cmd)
    stopped = False
    while p.poll() is None:
        time.sleep(30)
        if a.stop_file and not stopped and dt.datetime.now() >= end - dt.timedelta(minutes=5):
            open(a.stop_file, "w").write(json.dumps({"cmd": "stop", "reason": "window_gate: the away window closes in 5 minutes", "t": time.time()}))   # trainkit's CONTROL contract
            stopped = True
            log("window closes in 5 minutes: wrote stop to " + a.stop_file)
    log(f"finished with exit code {p.returncode}")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())

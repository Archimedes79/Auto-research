"""Experiment launcher: one command, one run record.  Standard library only (works in any python of the WSL).

    python tools/experiment/run.py --id E209.a --hyp "builder without night/blur rows still holds on night/blur" --basis E142 --seed 0 \
                  [--gpu-max-mb 9000 --wait-gpu 30] [--retries 2] [--result runs/e17/e142_b2/metrics.json] -- bash tools/env/_py14.sh <script.py> args   (or any command)

What it does (the loop-17 lessons, written down as code):
  * run record  experiment_execution/runs/<id>/run.json  (command, start / end, host, env, seed, status, exit code, duration, last log lines)
  * full log    experiment_execution/runs/<id>/run.log   (stdout + stderr of the command)
  * ledger row  experiment_execution/LEDGER.csv           (one row per run id, updated at the end; render with ledger.py)
  * the flag file  experiment_execution/runs/<id>/done  is written ONLY when the command exits with 0   (a crashed training must never leave a done flag)
  * GPU guard   waits until the GPU memory in use is below --gpu-max-mb (max --wait-gpu minutes) before it starts
  * retry       a run that dies with a CUDA 'unknown error' is repeated (--retries, default 2) after 15 s; any other failure stops at once
  * a run that wants to resume from a checkpoint finds the same --id again: the previous attempt is kept as run.<n>.log
Exit code = exit code of the command (1 if the GPU never became free).
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXEC = os.path.join(ROOT, "experiment_execution")
LEDGER = os.path.join(EXEC, "LEDGER.csv")
COLS = ["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes"]


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return None


def ledger_write(row):
    rows = []
    if os.path.exists(LEDGER):
        with open(LEDGER, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["id"] != row["id"]] + [{c: row.get(c, "") for c in COLS}]
    os.makedirs(EXEC, exist_ok=True)
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="run id, e.g. E209.a (experiment id + letter for the attempt / arm)")
    ap.add_argument("--hyp", default="", help="one-sentence hypothesis")
    ap.add_argument("--basis", default="", help="parent run / experiment ids")
    ap.add_argument("--seed", default="")
    ap.add_argument("--gpu-max-mb", type=int, default=0, help="start only when GPU memory used is below this (0 = no guard)")
    ap.add_argument("--wait-gpu", type=int, default=30, help="minutes to wait for the guard")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--result", default="", help="json file the command writes its metrics to; embedded in the ledger when it exists")
    ap.add_argument("--cwd", default=ROOT, help="working directory of the command (default: project root)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command (put it after --)")
    d = os.path.join(EXEC, "runs", a.id)
    os.makedirs(d, exist_ok=True)
    for old in ("done",):
        if os.path.exists(os.path.join(d, old)):
            os.remove(os.path.join(d, old))
    n = 1
    while os.path.exists(os.path.join(d, f"run.{n}.log")):
        n += 1
    if os.path.exists(os.path.join(d, "run.log")):
        os.rename(os.path.join(d, "run.log"), os.path.join(d, f"run.{n}.log"))
    rec = dict(id=a.id, hypothesis=a.hyp, basis=a.basis, seed=a.seed, command=" ".join(cmd), cwd=a.cwd, host=platform.node(), python=sys.version.split()[0],
               conda_env=os.environ.get("CONDA_DEFAULT_ENV", ""), started=dt.datetime.now().isoformat(timespec="seconds"), status="waiting")
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    if a.gpu_max_mb:
        t0 = time.time()
        while True:
            u = gpu_used_mb()
            if u is None or u < a.gpu_max_mb:
                break
            if time.time() - t0 > a.wait_gpu * 60:
                rec.update(status="gpu_busy", ended=dt.datetime.now().isoformat(timespec="seconds"), notes=f"GPU memory {u} MB in use for {a.wait_gpu} min")
                json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
                ledger_write(dict(rec, exit=1, duration_s=0))
                print("GPU did not become free:", rec["notes"])
                return 1
            time.sleep(30)
    rec["status"] = "running"
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    t1 = time.time()
    rc, attempt, tail = 1, 0, ""
    while attempt <= a.retries:
        attempt += 1
        with open(os.path.join(d, "run.log"), "a", encoding="utf-8", errors="replace") as lg:
            lg.write(f"=== attempt {attempt} {dt.datetime.now().isoformat(timespec='seconds')}: {' '.join(cmd)}\n")
            lg.flush()
            p = subprocess.run(cmd, cwd=a.cwd, stdout=lg, stderr=subprocess.STDOUT)
        rc = p.returncode
        tail = open(os.path.join(d, "run.log"), encoding="utf-8", errors="replace").read()[-3000:]
        if rc == 0 or "CUDA error" not in tail:
            break
        if attempt <= a.retries:
            time.sleep(15)
    dur = round(time.time() - t1)
    res = ""
    if a.result and os.path.exists(a.result):
        try:
            res = json.dumps(json.load(open(a.result)))[:400]
        except Exception:
            res = "(result file unreadable)"
    rec.update(status="ok" if rc == 0 else "failed", exit=rc, duration_s=dur, attempts=attempt, ended=dt.datetime.now().isoformat(timespec="seconds"), last_lines=tail.strip().splitlines()[-12:])
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    if rc == 0:
        open(os.path.join(d, "done"), "w").write(rec["ended"] + "\n")
    ledger_write(dict(rec, result=res, notes=f"{attempt} attempt(s)" if attempt > 1 else ""))
    print(f"[{a.id}] {rec['status']} (exit {rc}, {dur} s, {attempt} attempt(s)); log {os.path.join(d, 'run.log')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Experiment launcher: one command, one run record, one code tag.  Standard library only (works in any python of the WSL or Windows).

    python autoresearch/tools/run.py --id E240.a --hyp "..." --pred "Q >= 0.745" --rule "keep if > control + 0.010 on 2 seeds" --basis E219 --seed 0 \
           [--after E239.a,E239.c] [--gpu-max-mb 9000 --wait-gpu 30] [--retries 2] [--split test] [--result <metrics.json>] -- <command ...>

What it does (paths from the problem's project.json, see project.py):
  * run record  <run_records>/<id>/run.json  (command, start / end, host, env, seed, status, exit code, duration, commit, tag, last log lines)
  * full log    <run_records>/<id>/run.log   (stdout + stderr of the command)
  * ledger row  <ledger>                     (one row per run id; render with ledger.py)
  * done flag   <run_records>/<id>/done      ONLY when the command exits with 0 (a crashed training never leaves a done flag)
  * card check  warns when --hyp / --pred / --rule are missing (--require-card refuses)
  * code        commits the current code (the repository's whitelist decides what is code) and tags it <tag_prefix>/<id>; a resumed id keeps its
                first tag. --no-git skips this. The commit is written into the run record and the ledger.
  * after       --after <ids>: waits for the done flags of those runs (refuses when one of them failed); waiting is not counted as run time
  * TEST guard  --split test is refused unless the run id appears in the problem's TEST log
  * GPU guard   waits until the GPU memory in use is below --gpu-max-mb (max --wait-gpu minutes)
  * retry       a run that dies with a CUDA 'unknown error' is repeated (--retries) after 15 s; any other failure stops at once
  * resume      the same --id again keeps the previous attempt as run.<n>.log
Exit code = exit code of the command (1 if the GPU never became free, 2 if refused, 3 if a dependency failed).
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

COLS = ["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes", "prediction", "rule", "code_hash", "split", "commit"]


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def code_hash(root, cfg):
    """12-hex fingerprint of all .py / .sh of the code folders plus the extra files (works without git too)"""
    h = hashlib.sha1()
    for top in cfg["code_dirs"]:
        for dp, dn, fn in os.walk(os.path.join(root, top)):
            dn[:] = sorted(x for x in dn if x not in ("__pycache__", ".git"))
            for f in sorted(fn):
                if f.endswith((".py", ".sh")):
                    p = os.path.join(dp, f)
                    h.update(os.path.relpath(p, root).encode())
                    h.update(open(p, "rb").read())
    for extra in cfg["fingerprint_extra"]:
        p = os.path.join(root, extra)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def git(root, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", root, *args], capture_output=True, text=True, timeout=120)


def git_snapshot(root, prefix, run_id, msg):
    """commit the code (whitelist of the repository) and tag it prefix/run_id; returns (commit, tag, note)"""
    try:
        if git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return "no-git", "", ""
        tag = f"{prefix}/{run_id}"
        if git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0:  # resumed id: keep the first tag
            return git(root, "rev-list", "-n", "1", tag).stdout.strip()[:10], tag, "tag existed (resume)"
        for attempt in range(3):
            top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
            git(top, "add", "-A")
            if git(top, "diff", "--cached", "--quiet").returncode != 0:
                c = git(top, "commit", "-q", "-m", f"{run_id}: {msg}"[:200])
                if c.returncode != 0:
                    time.sleep(2)
                    continue
            t = git(top, "tag", "-a", tag, "-m", msg[:200] or run_id)
            if t.returncode == 0:
                return git(top, "rev-parse", "--short=10", "HEAD").stdout.strip(), tag, ""
            time.sleep(2)
        return "git-error", "", "commit / tag failed (index lock?)"
    except Exception as e:
        return "git-error", "", repr(e)[:120]


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return None


def ledger_write(ledger, row):
    rows = []
    if os.path.exists(ledger):
        with open(ledger, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    old = next((r for r in rows if r["id"] == row["id"]), {})  # a planned row keeps what was written before the run
    row = {**old, **{k: v for k, v in row.items() if v not in ("", None)}}
    rows = [r for r in rows if r["id"] != row["id"]] + [row]
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")  # unknown keys never break the write (loop-18 lesson)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])


def wait_after(records, ids, minutes):
    """wait for the done flags of ids; returns '' when all are done, else the reason"""
    t0 = time.time()
    while True:
        missing = []
        for i in ids:
            d = os.path.join(records, i)
            if os.path.exists(os.path.join(d, "done")):
                continue
            try:
                st = json.load(open(os.path.join(d, "run.json"))).get("status", "")
            except Exception:
                st = ""
            if st in ("failed", "gpu_busy", "blocked"):
                return f"dependency {i} {st}"
            missing.append(i)
        if not missing:
            return ""
        if time.time() - t0 > minutes * 60:
            return f"dependencies not done after {minutes} min: {', '.join(missing)}"
        time.sleep(30)


def main():
    root, cfg = P.root(), P.cfg()
    records, ledger, test_log = P.path("run_records"), P.path("ledger"), P.path("test_log")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="run id, e.g. E240.a (experiment id + letter for the arm / attempt)")
    ap.add_argument("--hyp", default="", help="one-sentence hypothesis")
    ap.add_argument("--pred", default="", help="prediction written BEFORE the run, a number (e.g. 'Q >= 0.745')")
    ap.add_argument("--rule", default="", help="decision rule written BEFORE the run")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="'test' needs a row naming this id in the TEST log")
    ap.add_argument("--require-card", action="store_true", help="refuse to start without --hyp, --pred and --rule")
    ap.add_argument("--basis", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--after", default="", help="comma list of run ids whose done flags this run waits for")
    ap.add_argument("--wait-after", type=int, default=720, help="minutes to wait for --after")
    ap.add_argument("--gpu-max-mb", type=int, default=0, help="start only when GPU memory used is below this (0 = no guard)")
    ap.add_argument("--wait-gpu", type=int, default=30)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--result", default="", help="json file the command writes its metrics to; embedded in the ledger when it exists")
    ap.add_argument("--no-git", action="store_true", help="do not commit and tag the code")
    ap.add_argument("--cwd", default=root, help="working directory of the command (default: problem folder)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command (put it after --)")
    missing = [n for n, v in (("--hyp", a.hyp), ("--pred", a.pred), ("--rule", a.rule)) if not v]
    if missing:
        print(f"[{a.id}] card incomplete, missing {', '.join(missing)} (hypothesis, prediction and decision rule are written before the run)")
        if a.require_card:
            return 2
    if a.split == "test":
        log = open(test_log, encoding="utf-8").read() if os.path.exists(test_log) else ""
        if not re.search(r"(?<![\w.])" + re.escape(a.id) + r"(?![\w.])", log):  # the whole id, not a prefix of a longer id
            print(f"[{a.id}] refused: split=test but {a.id} is not named in {os.path.relpath(test_log, root)} (log it first, with the person's OK)")
            return 2
    d = os.path.join(records, a.id)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "done")):
        os.remove(os.path.join(d, "done"))
    n = 1
    while os.path.exists(os.path.join(d, f"run.{n}.log")):
        n += 1
    if os.path.exists(os.path.join(d, "run.log")):
        os.rename(os.path.join(d, "run.log"), os.path.join(d, f"run.{n}.log"))
    rec = dict(id=a.id, hypothesis=a.hyp, prediction=a.pred, rule=a.rule, split=a.split, basis=a.basis, seed=a.seed, command=" ".join(cmd), cwd=a.cwd,
               host=platform.node(), python=sys.version.split()[0], conda_env=os.environ.get("CONDA_DEFAULT_ENV", ""), created=now(), status="waiting", after=a.after)
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    if a.after:
        why = wait_after(records, [x for x in a.after.split(",") if x], a.wait_after)
        if why:
            rec.update(status="blocked", ended=now(), notes=why)
            json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
            ledger_write(ledger, dict(rec, exit=3, duration_s=0))
            print(f"[{a.id}] blocked: {why}")
            return 3
    if a.gpu_max_mb:
        t0 = time.time()
        while True:
            u = gpu_used_mb()
            if u is None or u < a.gpu_max_mb:
                break
            if time.time() - t0 > a.wait_gpu * 60:
                rec.update(status="gpu_busy", ended=now(), notes=f"GPU memory {u} MB in use for {a.wait_gpu} min")
                json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
                ledger_write(ledger, dict(rec, exit=1, duration_s=0))
                print("GPU did not become free:", rec["notes"])
                return 1
            time.sleep(30)
    commit, tag, gnote = ("skipped", "", "") if a.no_git else git_snapshot(root, cfg["tag_prefix"], a.id, a.hyp or " ".join(cmd))
    rec.update(status="running", started=now(), code_hash=code_hash(root, cfg), commit=commit, tag=tag, git_note=gnote)
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    t1 = time.time()
    rc, attempt, tail = 1, 0, ""
    while attempt <= a.retries:
        attempt += 1
        with open(os.path.join(d, "run.log"), "a", encoding="utf-8", errors="replace") as lg:
            lg.write(f"=== attempt {attempt} {now()}: {' '.join(cmd)}\n")
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
    rec.update(status="ok" if rc == 0 else "failed", exit=rc, duration_s=dur, attempts=attempt, ended=now(), last_lines=tail.strip().splitlines()[-12:])
    json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
    if rc == 0:
        open(os.path.join(d, "done"), "w").write(rec["ended"] + "\n")
    notes = "; ".join(x for x in (f"{attempt} attempt(s)" if attempt > 1 else "", gnote) if x)
    ledger_write(ledger, dict(rec, result=res, notes=notes))
    print(f"[{a.id}] {rec['status']} (exit {rc}, {dur} s, {attempt} attempt(s)), code {commit} {tag}; log {os.path.join(d, 'run.log')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

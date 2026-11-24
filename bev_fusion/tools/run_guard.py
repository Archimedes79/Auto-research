"""run_guard.py - watches ONE long training run and stops it (`Training stop`: checkpoint, exit 75, HOLD) when a stop line is reached.
Promoted from the scratch script temp/t26/run_guard.py (card t26, checklist 11c; the scratch folder went by D140, 2026-11-23); the speed rule was rewritten (see below). Standard library only.

Stop lines (the values of card t26; each can be set):
  - free space of the disk (--disk, default /mnt/c = C: in the WSL) below --min-free-gb (6 GB),
  - seconds per step above --max-sps (14.4) on two looks in a row, from --min-step (500) on: the projection would pass the 96 h budget. The speed is the MEDIAN of the last 5 intervals
    between two logged training rows (log_every 20 steps: about the last 100 steps); an interval that touches a validation row (the evaluation time is in it) or whose steps do not
    increase (a restart that logs a step again) is skipped, and the median keeps one or two long intervals (a checkpoint save, the gap of a hold) from counting. Fewer than 5 such
    intervals: no speed yet. (The scratch version took the time of the last 100 steps as a whole: a hold gap or one long evaluation in that window stopped a resumed run.)
  - the validation row of step 3000 has radar_kept below 0.2 (the radar path collapsed; the t20 base reads 0.90); a marker file keeps a resumed run from being stopped again by the
    same row (a --dry look does not write it),
  - --stage-end N: the log shows step N (D102 stages: one guard start per stage; the lead reads the checkpoint, the owner says go on).
It only stops. It writes runs/_guards/<run id>.log and prints the same lines; 'GUARD STOP: <reason>' is what the lead tells the owner. A look that fails (a log row of an
unexpected kind, no interpreter, `Training stop` returning non-zero) is said and tried again at the next look (3 stop attempts), it never ends the guard silently. Without a log.jsonl
10 minutes after the start it says so on every look (a wrong run folder looks like a run that has not started).
Exit code: 0 the run finished (done flag) or a --dry look saw nothing to stop; 3 a stop line was reached (and, without --dry, the stop request went through); 4 the stop request failed 3 times,
or --max-failed-looks looks in a row failed (an hour at the default). The guard does not watch itself: the log gets a line at every look, so the modification time of runs/_guards/<id>.log is
at most 5 minutes old while it lives; look at it at every wake-up (a WSL restart kills the guard).
It ends by itself when the run's done flag (records/runs/<run id>/done) exists. HOLD stays until the owner's words clear it (`Training resume ... --words`).

    mkdir -p runs/_guards && setsid nohup python3 tools/run_guard.py t26.one_s0 --stage-end 4000 > runs/_guards/t26.one_s0.out 2>&1 &     (WSL, from bev_fusion; the run folder is runs/<id with . -> _>)
    python3 tools/run_guard.py t26.one_s0 --stage-end 200 --dry                                                 one look, the checks are made and said, nothing is stopped
Environment defaults (the names of the scratch version): GUARD_STAGE_END, GUARD_MIN_STEP, GUARD_EVERY_S, GUARD_DRY=1; GUARD_PYTHON = the interpreter of the stop request (default
~/miniconda3/envs/gcar/bin/python). Test: tests/test_run_guard.py (`--mutants`: the planted bugs)"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INTERVALS = 5               # the speed is the median of this many clean intervals between two logged training rows
STOP_TRIES = 3


def num(x):
    """a finite number (not a bool, not NaN)"""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def default_run_dir(run_id):
    """runs/<id with . as _> (what autoresearch/tools/run.py run_folder and the Training composer make of an id)"""
    return os.path.join("runs", run_id.replace(".", "_"))


def read_rows(path):
    """the rows of a log.jsonl that parse (a half-written last line or a stray line is skipped); [] when there is no file"""
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
    return rows


def clean_speeds(tr, val_steps, want=INTERVALS):
    """seconds per step of the last `want` intervals between two consecutive training rows `tr`, newest first, without the intervals that touch a validation step (a.step <= v <= b.step)
    and without those where the step or the time does not increase"""
    out = []
    j = len(tr) - 1
    while j > 0 and len(out) < want:
        a, b = tr[j - 1], tr[j]
        j -= 1
        if b["step"] <= a["step"] or b["t"] <= a["t"]:
            continue
        if any(a["step"] <= v <= b["step"] for v in val_steps):
            continue
        out.append((b["t"] - a["t"]) / (b["step"] - a["step"]))
    return out


def decide(rows, free_gb, slow, rk_seen, stage_end=0, min_step=500, max_sps=14.4, min_free_gb=6.0, rk_min=0.2):
    """one look at the log -> (reason or None, slow looks so far, radar_kept stop taken, note). The order of the lines is fixed: free space, radar_kept, stage end, speed.
    `slow` is the number of slow looks in a row before this one; `rk_seen` says the radar_kept stop was taken before (marker). A training row has "t" (seconds) and "step"
    and no "val"; a validation row has "val" and "step"; a row with a value of another kind (null, a string) is skipped."""
    if free_gb < min_free_gb:
        return f"C: free space {free_gb:.1f} GB is below {min_free_gb:g} GB", slow, False, ""
    tr = [r for r in rows if "val" not in r and num(r.get("t")) and num(r.get("step"))]
    va = [r for r in rows if "val" in r and num(r.get("step"))]
    warn = ""
    for r in va:
        val = r["val"] if isinstance(r["val"], dict) else {}
        kept = val.get("radar_kept", 1.0)
        if r["step"] == 3000 and not num(kept):
            warn = f"WARNING: radar_kept at step 3000 is {kept!r}, not a number: the radar line cannot be read"
        elif r["step"] == 3000 and kept < rk_min and not rk_seen:
            return f"radar_kept {kept:.3f} at step 3000 is below {rk_min:g} (the radar path collapsed)", slow, True, ""
    if stage_end and tr and tr[-1]["step"] >= stage_end:
        return f"stage end: step {tr[-1]['step']} reached the stage end {stage_end} (D102; the lead reads the checkpoint, the owner says go on)", slow, False, warn
    if tr and tr[-1]["step"] >= min_step:
        last = tr[-1]
        sp = clean_speeds(tr, [r["step"] for r in va])
        if len(sp) < INTERVALS:
            return None, slow, False, f"step {last['step']}: fewer than {INTERVALS} clean intervals logged, no speed yet" + (" | " + warn if warn else "")
        sps = statistics.median(sp)
        slow = slow + 1 if sps > max_sps else 0
        note = f"step {last['step']}: {sps:.1f} s per step (median of the last {INTERVALS} clean intervals), C: free {free_gb:.1f} GB, slow looks {slow}" + (" | " + warn if warn else "")
        if slow >= 2:
            return f"{sps:.1f} s per step on two looks in a row: the projection passes the 96 h budget", slow, False, note
        return None, slow, False, note
    return None, slow, False, f"waiting: {len(tr)} rows, C: free {free_gb:.1f} GB" + (" | " + warn if warn else "")


def poll_seconds(stage_end, last_step, every, near_every):
    """seconds to the next look: `near_every` in the last 60 steps before a stage end, else `every`"""
    return near_every if stage_end and last_step is not None and stage_end - last_step <= 60 else every


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", help="the run id of the ledger (records/runs/<id>/done is its done flag)")
    ap.add_argument("--run-dir", help="the run folder (default runs/<id with . -> _>)")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "runs", "_guards"))
    ap.add_argument("--stage-end", type=int, default=int(os.environ.get("GUARD_STAGE_END", "0")))
    ap.add_argument("--min-step", type=int, default=int(os.environ.get("GUARD_MIN_STEP", "500")))
    ap.add_argument("--every", type=int, default=int(os.environ.get("GUARD_EVERY_S", "300")), help="seconds between two looks")
    ap.add_argument("--near-every", type=int, default=15, help="seconds between two looks in the last 60 steps before a stage end")
    ap.add_argument("--max-sps", type=float, default=14.4)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--records-dir", default=os.path.join(ROOT, "records", "runs"), help="where records/runs/<id>/done lives")
    ap.add_argument("--max-failed-looks", type=int, default=12, help="the guard gives up (exit 4) after this many failed looks in a row (an hour at the default)")
    ap.add_argument("--no-log-warn-s", type=int, default=600, help="seconds after the start from which a missing log is warned about on every look")
    ap.add_argument("--disk", default="/mnt/c" if os.path.isdir("/mnt/c") else ROOT, help="the disk whose free space is watched (default C: in the WSL)")
    ap.add_argument("--dry", action="store_true", default=os.environ.get("GUARD_DRY") == "1", help="one look, nothing is stopped, no marker is written")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = os.path.join(ROOT, a.run_dir or default_run_dir(a.run_id))
    log_file, done = os.path.join(run_dir, "log.jsonl"), os.path.join(a.records_dir, a.run_id, "done")
    os.makedirs(a.log_dir, exist_ok=True)
    out = os.path.join(a.log_dir, a.run_id + ".log")
    marker = out + ".rk"

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        try:
            with open(out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:                                                 # a full disk must not kill the guard: the line is on stdout
            pass

    def stop(reason):
        """-> True when the stop request went through (or a --dry look)"""
        say(f"GUARD STOP: {reason}")
        if a.dry:
            say("(dry look: no stop request sent)")
            return True
        env = dict(os.environ, AR_PROJECT=os.path.join("..", os.path.basename(ROOT)), PYTHONPATH=f"{os.path.dirname(ROOT)}:{os.path.join(os.path.dirname(ROOT), 'autoresearch')}")
        py = os.environ.get("GUARD_PYTHON") or os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
        cmd = [py, "-m", "Training", "stop", a.run_id, "--reason", reason, "--wait", "900"]
        try:
            r = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(ROOT), "autoresearch"), env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=1200)
        except (OSError, subprocess.SubprocessError) as e:
            say(f"stop request FAILED ({type(e).__name__}: {str(e)[:200]})")
            return False
        say(f"stop request rc={r.returncode}: {(r.stdout or r.stderr)[-300:]}")
        return r.returncode == 0

    t0 = time.time()
    say(f"guard started for {a.run_id}, every {a.every} s" + (", dry" if a.dry else "") + f"; log {log_file} (exists: {os.path.exists(log_file)}), disk {a.disk}")
    slow, tries, fails = 0, 0, 0
    while True:
        if os.path.exists(done):
            say("run finished (done flag): guard ends")
            return 0
        reason, rk_hit, tr_last = None, False, None
        try:
            rows = read_rows(log_file)
            reason, slow, rk_hit, note = decide(rows, shutil.disk_usage(a.disk).free / 1e9, slow, os.path.exists(marker), a.stage_end, a.min_step, a.max_sps, a.min_free_gb)
            tr = [r for r in rows if "val" not in r and num(r.get("t")) and num(r.get("step"))]
            tr_last = tr[-1]["step"] if tr else None
            if note:
                say(note)
            if not rows and time.time() - t0 > a.no_log_warn_s:
                say(f"WARNING: no rows in {log_file} {int(time.time() - t0)} s after the start: a wrong run folder (--run-dir), or the run has not started")
        except Exception as e:                                          # a look must not end the guard: say it, look again
            fails += 1
            say(f"look FAILED ({type(e).__name__}: {str(e)[:200]}), looking again at the next look ({fails} of {a.max_failed_looks} failed in a row)")
            if a.dry:
                return 1
            if fails >= a.max_failed_looks:
                say(f"GUARD GIVES UP: {fails} looks in a row failed, the run is NOT guarded any more")
                return 4
        else:
            fails = 0
        if reason:
            if stop(reason):
                if rk_hit and not a.dry:                                # the marker only after the stop went through, so that a failed stop is tried again
                    with open(marker, "w", encoding="utf-8") as f:
                        f.write("seen")
                return 3
            tries += 1
            if tries >= STOP_TRIES:
                say(f"GUARD GIVES UP: the stop request failed {tries} times, the run is NOT guarded any more")
                return 4
        if a.dry:
            return 0
        time.sleep(poll_seconds(a.stage_end, tr_last, a.every, a.near_every))


if __name__ == "__main__":
    sys.exit(main())

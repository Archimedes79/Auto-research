"""Test of tools/run_guard.py (the stop lines of a long run): known answers of `decide` at and around every limit, the speed rule on a resumed run and on an evaluation window, the log
reader, the command line (dry look, stop request through a stub interpreter, failing stops, done flag, wrong disk, missing log) and the defaults.
Standard library only; no GPU, no run; seconds. Run from bev_fusion: python tests/test_run_guard.py
`--mutants`: plants the bugs of MUTANTS one by one into a copy of the guard; every one must make this test fail (an equivalent change is listed under EQUIVALENT)."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RG_DIR = os.environ.get("RUN_GUARD_DIR") or os.path.join(ROOT, "tools")        # a mutant copy of the guard when set
sys.path.insert(0, RG_DIR)
import run_guard as G  # noqa: E402

# (name, old text in tools/run_guard.py, planted text): each must be found exactly once and make the test fail
MUTANTS = [
    ("free space: <= instead of <", "if free_gb < min_free_gb:", "if free_gb <= min_free_gb:"),
    ("two looks: one is enough", "if slow >= 2:", "if slow >= 1:"),
    ("a fast look does not reset", "slow = slow + 1 if sps > max_sps else 0", "slow = slow + 1 if sps > max_sps else slow"),
    ("speed: >= instead of >", "slow + 1 if sps > max_sps", "slow + 1 if sps >= max_sps"),
    ("radar_kept: any step from 3000 on", 'elif r["step"] == 3000 and kept < rk_min', 'elif r["step"] >= 3000 and kept < rk_min'),
    ("radar_kept: <= instead of <", "kept < rk_min", "kept <= rk_min"),
    ("radar_kept: no marker check", "and not rk_seen", ""),
    ("stage end: > instead of >=", 'tr[-1]["step"] >= stage_end', 'tr[-1]["step"] > stage_end'),
    ("stage end read after the speed", 'if stage_end and tr and tr[-1]["step"] >= stage_end:', "if False:"),
    ("min step: > instead of >=", 'tr[-1]["step"] >= min_step', 'tr[-1]["step"] > min_step'),
    ("one interval is a speed", "INTERVALS = 5", "INTERVALS = 1"),
    ("mean instead of median", "statistics.median(sp)", "statistics.mean(sp)"),
    ("intervals with a validation row count", 'if any(a["step"] <= v <= b["step"] for v in val_steps):', "if False:"),
    ("a validation row touches only one side", 'a["step"] <= v <= b["step"]', 'a["step"] < v <= b["step"]'),
    ("intervals that do not advance count", 'if b["step"] <= a["step"] or b["t"] <= a["t"]:', "if False:"),
    ("a dry look writes the marker", "if rk_hit and not a.dry:", "if rk_hit:"),
    ("run folder without . -> _", 'run_id.replace(".", "_")', "run_id"),
    ("stop request: --wait 9", '"--wait", "900"', '"--wait", "9"'),
    ("default max-sps 15", 'default=14.4)', 'default=15.0)'),
    ("default min-free-gb 5", '"--min-free-gb", type=float, default=6.0', '"--min-free-gb", type=float, default=5.0'),
    ("default look every 30 s", '"GUARD_EVERY_S", "300"', '"GUARD_EVERY_S", "30"'),
    ("default min step 50", '"GUARD_MIN_STEP", "500"', '"GUARD_MIN_STEP", "50"'),
    ("done flag named DONE", 'os.path.join(a.records_dir, a.run_id, "done")', 'os.path.join(a.records_dir, a.run_id, "DONE")'),
    ("one stop try", "STOP_TRIES = 3", "STOP_TRIES = 1"),
    ("a reached stop line exits 0", "                return 3\n", "                return 0\n"),
    ("polling: near window 600 steps", "stage_end - last_step <= 60", "stage_end - last_step <= 600"),
    ("polling: never near", "return near_every if stage_end", "return every if stage_end"),
    ("the disk argument is ignored", "shutil.disk_usage(a.disk)", "shutil.disk_usage(ROOT)"),
    ("the stop interpreter is ignored", 'os.environ.get("GUARD_PYTHON") or ', ""),
    ("failed looks: never give up", "if fails >= a.max_failed_looks:", "if False:"),
    ("default max-failed-looks 1", '"--max-failed-looks", type=int, default=12', '"--max-failed-looks", type=int, default=1'),
    ("a radar_kept that is no number is silent", 'if r["step"] == 3000 and not num(kept):', "if False:"),
]
EQUIVALENT = ["the 'val' not in r test of the training rows (a validation row has no t)", "the 1200 s timeout of the stop request",
              "a failed look in a real (not dry) guard that ends the guard instead of looking again (it would need an endless loop to show)",
              "untested, not planted: a good look resetting the counter of failed looks in a row (it needs a flapping failure), the OSError of a full log disk in say(), non-UTF-8 output of the stop request"]


def train_rows(first, last, sec_per_step, t0=1000.0, every=1):
    """training rows step first..last (every `every`th), time stamps t0 + sec_per_step * (step - first)"""
    return [{"step": s, "t": t0 + sec_per_step * (s - first), "total": 20.0} for s in range(first, last + 1, every)]


def look(rows, free=50.0, slow=0, rk=False, **kw):
    return G.decide(rows, free, slow, rk, **kw)


val = lambda step, kept=0.9: {"step": step, "val": {"radar_kept": kept, "total": 25.0}}      # noqa: E731


def known_answers():
    # 1 a quiet run: 9.4 s per step at step 600, nothing to stop, the speed is said
    reason, slow, rk_hit, note = look(train_rows(1, 601, 9.4, every=20))
    assert reason is None and slow == 0 and not rk_hit and "9.4 s per step" in note, (reason, note)
    print("  ok   a run at 9.4 s per step is not stopped, the look says the speed")

    # 2 free space: below 6 GB stops (also with no log yet), exactly 6 does not
    rows = train_rows(1, 601, 9.4, every=20)
    assert "C: free space 5.9 GB" in look(rows, free=5.9)[0] and look(rows, free=6.0)[0] is None and look([], free=1.0)[0] is not None
    print("  ok   free space below 6 GB stops (also with no log yet), exactly 6 does not")

    # 3 speed: one slow look is not enough, two in a row stop, a fast look resets, the limit itself is not slow
    slow_rows = train_rows(1, 701, 15.0, every=20)
    r1 = look(slow_rows)
    assert r1[0] is None and r1[1] == 1 and "15.0 s per step" in r1[3], r1
    r2 = look(slow_rows, slow=r1[1])
    assert r2[0] is not None and "two looks in a row" in r2[0] and r2[1] == 2, r2
    fast_rows = train_rows(1, 701, 9.0, every=20)
    assert look(fast_rows, slow=1)[1] == 0 and look(fast_rows, slow=1)[0] is None
    edge = train_rows(1, 701, 14.5, every=20)                          # 14.5 is exact in binary: the limit itself is not slow
    assert look(edge, slow=1, max_sps=14.5)[0] is None and look(edge, slow=1, max_sps=14.5)[1] == 0 and look(edge, slow=1, max_sps=14.4)[0] is not None
    print("  ok   two slow looks in a row stop; one does not; a fast look resets; the limit itself is not slow")

    # 4 the speed is read from step 500 on and needs 5 clean intervals behind it; it is the LAST intervals, not the mean of the run
    assert look(train_rows(1, 481, 30.0, every=20))[0] is None and "waiting" in look(train_rows(1, 481, 30.0, every=20))[3]
    assert look(train_rows(1, 501, 30.0, every=20))[1] == 1 and look(train_rows(1, 500, 30.0, every=20))[1] == 0       # step 501 counts, 481 does not
    five = train_rows(496, 500, 30.0)                                  # 4 intervals only
    assert look(five)[0] is None and look(five)[1] == 0 and "no speed yet" in look(five)[3]
    mixed = train_rows(1, 601, 9.0, every=20) + [{"step": s, "t": 1000.0 + 9.0 * 600 + 30.0 * (s - 601)} for s in range(621, 721, 20)]       # fast for 600 steps, slow in the last 100
    assert look(mixed)[1] == 1, "the speed is the last intervals, not the mean of the run"
    print("  ok   the speed is the median of the last 5 intervals, read from step 500, five intervals needed")

    # 5 radar_kept at step 3000: below 0.2 stops once; 0.2, other steps, no value, a null or a string do not; a marker (seen before) does not
    base = train_rows(1, 3001, 9.4, every=20)
    r = look(base + [val(3000, 0.1)])
    assert r[0] and "radar_kept 0.100 at step 3000" in r[0] and r[2] is True, r
    assert look(base + [val(3000, 0.2)])[0] is None and look(base + [val(3000, 0.9)])[0] is None
    assert look(base + [val(2999, 0.0)])[0] is None and look(base + [val(3001, 0.0)])[0] is None
    assert look(base + [val(3000, 0.1)], rk=True)[0] is None, "a resumed run is not stopped again by the same row"
    for odd in ({"step": 3000, "val": {"total": 25.0}}, {"step": 3000, "val": 0.5}, {"step": "3000", "val": {"radar_kept": 0.0}}):
        assert look(base + [odd])[0] is None and "not a number" not in look(base + [odd])[3], odd            # no value, no dict, no step: nothing to say
    for odd in ({"step": 3000, "val": {"radar_kept": None}}, {"step": 3000, "val": {"radar_kept": "x"}}, {"step": 3000, "val": {"radar_kept": float("nan")}}):
        got = look(base + [odd])
        assert got[0] is None and "WARNING: radar_kept at step 3000" in got[3] and "not a number" in got[3], (odd, got)    # a value that cannot be read is said, not silent
    print("  ok   radar_kept below 0.2 at step 3000 stops once (marker); 0.2, other steps do not; a value that is no number is warned about, not a crash")

    # 6 stage end: stops at its step, not before; 0 = none; a validation row is no training step; rows of another kind are skipped
    rows4 = train_rows(1, 3981, 9.4, every=20) + [{"step": 4000, "t": 1000.0 + 9.4 * 3999}]
    assert "stage end: step 4000 reached the stage end 4000" in look(rows4, stage_end=4000)[0]
    assert look(rows4, stage_end=4001)[0] is None and look(rows4, stage_end=0)[0] is None
    assert look(train_rows(1, 3981, 9.4, every=20), stage_end=4000)[0] is None
    assert look([val(4000, 0.9)], stage_end=4000)[0] is None, "a validation row is not a training step"
    assert look(rows4 + [{"step": 4020, "t": None}, {"step": None, "t": 5.0}, {"t": "x", "step": 4040}], stage_end=4021)[0] is None
    print("  ok   the stage end stops at its step, not before; 0 means none; validation and odd rows do not count")

    # 7 the order: free space before radar_kept before the stage end before the speed
    both = train_rows(1, 4001, 15.0, every=20) + [val(3000, 0.0)]
    assert "free space" in look(both, free=1.0, stage_end=4000)[0]
    assert "radar_kept" in look(both, stage_end=4000)[0]
    assert "stage end" in look(train_rows(1, 4001, 15.0, every=20), stage_end=4000, slow=1)[0]
    print("  ok   the lines are read in the fixed order: free space, radar_kept, stage end, speed")

    # 8 a RESUMED run: a hold of 6 h sits in the log; the fresh guard of the next stage must not read it as a slow step (the scratch version stopped such a run on its second look)
    before = train_rows(1, 4001, 10.0, every=20)
    after = [{"step": s, "t": 1000.0 + 10.0 * 4000 + 21600.0 + 10.0 * (s - 4000)} for s in range(4020, 4121, 20)]
    for upto in (2, 3, 4, 6):                                          # looks right after the resume, with 2 .. 6 rows behind the gap
        reason, slow, _, note = look(before + after[:upto], min_step=500)
        assert reason is None and slow == 0, (upto, reason, note)
    first = look(before + after[:2])
    second = look(before + after[:3], slow=first[1])
    assert second[0] is None and "10.0 s per step" in second[3], second
    print("  ok   a resumed run (6 h gap in the log) is not read as slow, also on the second look of a fresh guard")

    # 9 an EVALUATION in the window: intervals that touch a validation row are skipped, also three evaluations in a row; a long evaluation does not make a slow step
    ev = train_rows(1, 6101, 10.5, every=20)
    for k in range(len(ev)):
        if ev[k]["step"] > 6000:
            ev[k] = dict(ev[k], t=ev[k]["t"] + 2000.0)                 # 2000 s of evaluation in the interval 6000 -> 6020, carried by every later row
    assert look(ev + [val(6000)])[0] is None and look(ev + [val(6000)])[1] == 0, look(ev + [val(6000)])
    ev3 = train_rows(1, 6101, 10.5, every=20)
    for k in range(len(ev3)):                                          # three evaluations in the last 5 intervals: 6040, 6060, 6080
        ev3[k] = dict(ev3[k], t=ev3[k]["t"] + sum(2000.0 for e in (6040, 6060, 6080) if ev3[k]["step"] > e))
    assert look(ev3 + [val(6040), val(6060), val(6080)])[1] == 0 and "10.5 s per step" in look(ev3 + [val(6040), val(6060), val(6080)])[3]
    assert look(ev3)[1] == 1, "without the validation rows the same log reads slow (the rows are what excludes the intervals)"
    side = train_rows(1, 6101, 10.5, every=20)                          # a validation step EQUAL to the first step of an interval (6001, 6041, 6081 are logged steps): that interval is excluded too
    for k in range(len(side)):
        side[k] = dict(side[k], t=side[k]["t"] + sum(3000.0 for e in (6001, 6041, 6081) if side[k]["step"] > e))
    assert look(side + [val(6001), val(6041), val(6081)])[1] == 0 and look(side)[1] == 1
    print("  ok   an evaluation in the window is not a slow step (intervals that touch a validation row are skipped)")

    # 10 outliers: one or two long intervals (a checkpoint, a gap) do not count, three of five do; a restart that logs a step again is skipped
    out2 = train_rows(1, 701, 10.0, every=20)
    for k in (-2, -4):
        out2[k] = dict(out2[k], t=out2[k]["t"] + 5000.0)
        out2[k + 1:] = [dict(r, t=r["t"] + 5000.0) for r in out2[k + 1:]]
    assert look(out2)[1] == 0, look(out2)
    out3 = train_rows(1, 701, 10.0, every=20)
    for k in (-2, -4, -5):
        out3[k:] = [dict(r, t=r["t"] + 5000.0) for r in out3[k:]]
    assert look(out3)[1] == 1, look(out3)
    redo = train_rows(1, 701, 10.0, every=20) + [{"step": 681, "t": 9999.0}, {"step": 701, "t": 10200.0}, {"step": 721, "t": 10400.0}, {"step": 721, "t": 10400.0}]       # the last row logged twice: step 0 apart
    assert look(redo)[0] is None and "10.0 s per step" in look(redo)[3], look(redo)
    print("  ok   one or two long intervals do not count, three of five do; a step logged again after a restart is skipped")


def reader_and_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "log.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "t": 1.0}) + "\nnot json\n[1, 2]\n" + json.dumps({"step": 2, "t": 2.0}) + '\n{"step": 3, "t"')
        assert [r["step"] for r in G.read_rows(p)] == [1, 2], G.read_rows(p)
        assert G.read_rows(os.path.join(tmp, "missing.jsonl")) == []
    print("  ok   the log reader skips stray and half-written lines; a missing log is empty")
    a = G.build_parser().parse_args(["x.y"])
    for k, v in (("max_sps", 14.4), ("min_free_gb", 6.0), ("every", 300), ("min_step", 500), ("stage_end", 0), ("near_every", 15), ("no_log_warn_s", 600), ("max_failed_looks", 12), ("dry", False)):
        if k in ("every", "min_step", "stage_end", "dry") and any(os.environ.get(e) for e in ("GUARD_EVERY_S", "GUARD_MIN_STEP", "GUARD_STAGE_END", "GUARD_DRY")):
            continue                                                    # the caller set an environment default
        assert getattr(a, k) == v, (k, getattr(a, k), v)
    assert G.default_run_dir("t26.one_s0") == os.path.join("runs", "t26_one_s0") and G.default_run_dir("a.b.c") == os.path.join("runs", "a_b_c")
    try:                                                                # the mapping is the one run.py makes (the autoresearch launcher): checked when the module imports here
        sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "autoresearch", "tools"))
        import run as R
        assert R.run_folder("t26.one_s0") == "t26_one_s0" and R.run_folder("a.b.c") == "a_b_c"
        mapped = "against run.py run_folder"
    except ImportError:
        mapped = "run.py not importable here"
    assert G.poll_seconds(4000, 3950, 300, 15) == 15 and G.poll_seconds(4000, 3939, 300, 15) == 300 and G.poll_seconds(0, 3950, 300, 15) == 300 and G.poll_seconds(4000, None, 300, 15) == 300
    assert G.poll_seconds(4000, 3940, 300, 15) == 15 and G.poll_seconds(4000, 4000, 300, 15) == 15
    print(f"  ok   the defaults (14.4 s, 6 GB, 300 s, step 500, 15 s near a stage end), the run folder mapping ({mapped}), the polling interval")


def command_line():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir, logs, recs = os.path.join(tmp, "run"), os.path.join(tmp, "guards"), os.path.join(tmp, "records")
        os.makedirs(run_dir)

        def write_log(rows):
            with open(os.path.join(run_dir, "log.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        def cli(*extra, env=None, run_id="fake.guard"):
            e = dict(os.environ, **(env or {}))
            c = subprocess.run([sys.executable, os.path.join(RG_DIR, "run_guard.py"), run_id, "--run-dir", run_dir, "--log-dir", logs, "--records-dir", recs, "--min-step", "100",
                                "--min-free-gb", "0", "--every", "1", "--near-every", "1", *extra], capture_output=True, text=True, timeout=120, env=e)
            return c.returncode, c.stdout + c.stderr

        write_log(train_rows(1, 301, 9.4, every=20))
        # a dry look: the stage end is reached (rc 3, nothing sent), not reached (rc 0, the speed is said); the log file is written
        rc, out = cli("--stage-end", "200", "--dry")
        assert rc == 3 and "GUARD STOP: stage end: step 301 reached the stage end 200" in out and "dry look: no stop request sent" in out, (rc, out)
        rc, out = cli("--stage-end", "400", "--dry")
        assert rc == 0 and "GUARD STOP" not in out and "9.4 s per step" in out and "exists: True" in out, (rc, out)
        assert os.path.exists(os.path.join(logs, "fake.guard.log"))
        print("  ok   a dry look stops at a reached stage end (exit 3, nothing sent), says the speed when it is not reached (exit 0), and writes its log")

        # the done flag ends the guard
        os.makedirs(os.path.join(recs, "fake.guard"))
        open(os.path.join(recs, "fake.guard", "done"), "w").close()
        rc, out = cli("--stage-end", "200", "--dry")
        assert rc == 0 and "run finished (done flag)" in out and "GUARD STOP" not in out, (rc, out)
        os.remove(os.path.join(recs, "fake.guard", "done"))
        print("  ok   the done flag of the run record ends the guard")

        # the disk argument is the one measured; a missing log is said; a wrong disk is a failed look, not a crash
        rc, out = cli("--dry", "--disk", os.path.join(tmp, "no_such_disk"))
        assert rc == 1 and "look FAILED" in out, (rc, out)
        rc, out = cli("--disk", os.path.join(tmp, "no_such_disk"), "--max-failed-looks", "2")           # a permanent failure ends the guard visibly, after the set number of looks
        assert rc == 4 and out.count("look FAILED") == 2 and "GUARD GIVES UP: 2 looks in a row failed" in out, (rc, out)
        os.remove(os.path.join(run_dir, "log.jsonl"))
        rc, out = cli("--dry", "--no-log-warn-s", "-1")
        assert rc == 0 and "exists: False" in out and "WARNING: no rows in" in out and "waiting: 0 rows" in out, (rc, out)
        print("  ok   the --disk argument is the one measured (a wrong one is a failed look); a missing log is said at the start and warned about")

        if os.name == "nt":
            print("  skip the stop request tests (no sh on Windows)")
            return
        # a real stop request through a stub interpreter that records what it was called with
        stub, rec = os.path.join(tmp, "stub.sh"), os.path.join(tmp, "stub.out")
        with open(stub, "w") as f:
            f.write('#!/bin/sh\necho "args: $@" >> "$STUB_OUT"\necho "cwd: $(pwd)" >> "$STUB_OUT"\necho "AR_PROJECT: $AR_PROJECT" >> "$STUB_OUT"\necho "PYTHONPATH: $PYTHONPATH" >> "$STUB_OUT"\nexit ${STUB_RC:-0}\n')
        os.chmod(stub, 0o755)
        write_log(train_rows(1, 301, 9.4, every=20))
        rc, out = cli("--stage-end", "200", env={"GUARD_PYTHON": stub, "STUB_OUT": rec})
        got = open(rec).read()
        assert rc == 3 and "stop request rc=0" in out, (rc, out)
        assert "args: -m Training stop fake.guard --reason stage end: step 301" in got and "--wait 900" in got, got
        cwd_line = [ln for ln in got.splitlines() if ln.startswith("cwd: ")][0]
        assert cwd_line.replace("\\", "/").rstrip("/").endswith("/autoresearch"), got
        assert "AR_PROJECT: ../" + os.path.basename(ROOT) in got and "autoresearch" in got.split("PYTHONPATH: ")[1], got
        print("  ok   a real stop calls `Training stop <id> --reason .. --wait 900` from autoresearch/ with AR_PROJECT and PYTHONPATH, exit 3")

        # radar_kept: the marker is written only by a real stop that went through, never by a dry look, and a failed stop is tried again (3 tries, exit 4, no marker)
        write_log(train_rows(1, 3001, 9.4, every=20) + [val(3000, 0.05)])
        marker = os.path.join(logs, "fake.guard.log.rk")
        rc, out = cli("--dry", "--min-step", "100")
        assert rc == 3 and "radar_kept 0.050 at step 3000" in out and not os.path.exists(marker), (rc, out)
        os.remove(rec)
        rc, out = cli(env={"GUARD_PYTHON": stub, "STUB_OUT": rec, "STUB_RC": "1"})
        assert rc == 4 and out.count("stop request rc=1") == 3 and "GUARD GIVES UP" in out and not os.path.exists(marker), (rc, out)
        assert open(rec).read().count("args: ") == 3
        rc, out = cli(env={"GUARD_PYTHON": os.path.join(tmp, "no_such_python"), "STUB_OUT": rec})
        assert rc == 4 and out.count("stop request FAILED") == 3 and not os.path.exists(marker), (rc, out)
        rc, out = cli(env={"GUARD_PYTHON": stub, "STUB_OUT": rec})
        assert rc == 3 and os.path.exists(marker), (rc, out)
        rc, out = cli("--dry")
        assert rc == 0 and "GUARD STOP" not in out, "the marker keeps a resumed run from being stopped by the same row"
        print("  ok   radar_kept: a dry look writes no marker; a failing stop is tried 3 times (exit 4, no marker, also with no interpreter); a stop that went through writes it")


def mutants():
    src = open(os.path.join(ROOT, "tools", "run_guard.py"), encoding="utf-8").read()

    def run_copy(text):
        """the test against a copy of the guard that sits in a tree like the repository (tmp/bev_fusion/tools, tmp/autoresearch)"""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "bev_fusion", "tools"))
            os.makedirs(os.path.join(tmp, "autoresearch"))
            with open(os.path.join(tmp, "bev_fusion", "tools", "run_guard.py"), "w", encoding="utf-8") as f:
                f.write(text)
            return subprocess.run([sys.executable, os.path.join(HERE, "test_run_guard.py")], capture_output=True, text=True, timeout=600,
                                  env=dict(os.environ, RUN_GUARD_DIR=os.path.join(tmp, "bev_fusion", "tools")))
    clean = run_copy(src)
    assert clean.returncode == 0, "the unplanted copy must pass the test:\n" + clean.stdout[-600:] + clean.stderr[-600:]
    killed = 0
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant '{name}': the text to plant into is found {src.count(old)} times"
        ok = run_copy(src.replace(old, new)).returncode != 0
        killed += ok
        print(f"  {'caught  ' if ok else 'SURVIVED'} {name}")
    print(f"{killed} of {len(MUTANTS)} planted bugs caught; equivalent (not planted): " + "; ".join(EQUIVALENT))
    assert killed == len(MUTANTS), "a planted bug survived: the test does not guard that line"


def main():
    if "--mutants" in sys.argv:
        return mutants()
    known_answers()
    reader_and_defaults()
    command_line()
    print("run_guard test ok")


if __name__ == "__main__":
    main()

"""Self-test of the launcher and ledger rules of the deep review of 2026-11-10 (run.py, ledger.py, hold.py): the TEST gate, the command pid of a run whose launcher
was killed, the ledger freeze, id twins, the scratch scripts of a snapshot, notes that stay, the card hash, a hold that arrives before the start, the settle wait.

    python autoresearch/tools/selftest_launch.py        (needs git and bash, e.g. in the WSL; selftest.py runs it at its end)

Builds its own throw-away problem folder with its own git repository; the real ledger, run folders, queue and GPU are never touched.
"""
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as RUN  # noqa: E402

T = tempfile.mkdtemp(prefix="ar_launch_")
os.makedirs(f"{T}/code")
open(f"{T}/code/a.py", "w").write("x = 1\n")
open(f"{T}/aaa.json", "w").write("0\n")
CFG = dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md", code_dirs=["code"],
           fingerprint_extra=["aaa.json"], scratch="temp", live_code=["code"])
env = dict(os.environ, AR_PROJECT=T, AR_MACHINE_LOCK=f"{T}/state/machine_lock.json", AR_SETTLE_S="0")      # never the real machine lock
env.pop("AR_LOCK_OWNER", None)
G = ["git", "-c", "safe.directory=*", "-C", T]
subprocess.run(G[:3] + ["init", "-q", T], check=True)
for k, v in (("user.name", "selftest"), ("user.email", "selftest@example.invalid"), ("core.autocrlf", "false")):
    subprocess.run(G + ["config", k, v], check=True)
stale = dict(env, AR_STALE_S="1")
LANE_LOG = os.path.join(HERE, "..", "..", "bev_fusion_lane_center", "evaluation", "TEST_USAGE_LOG.md")


def set_cfg(**kw):
    json.dump(dict(CFG, **kw), open(f"{T}/project.json", "w"))


def go(rid, cmd, *extra, e=None):
    return subprocess.run([sys.executable, f"{HERE}/run.py", "--id", rid, "--hyp", "h", "--pred", "p", "--rule", "r", "--retries", "0", *extra, "--", *cmd],
                          capture_output=True, text=True, env=e or env)


def led(*args):
    return subprocess.run([sys.executable, f"{HERE}/ledger.py", *args], capture_output=True, text=True, env=env)


def rows():
    if not os.path.exists(f"{T}/LEDGER.csv"):
        return {}
    return {r["id"]: r for r in csv.DictReader(open(f"{T}/LEDGER.csv", newline="", encoding="utf-8"))}


def rec_of(rid):
    return RUN.record(f"{T}/runs/{rid}")


def background(rid, cmd, *extra):
    """a launcher in the background (its command is `cmd`); waits until the record says running and holds the command's pid, or the launcher's own status"""
    p = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", rid, "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", *extra, "--", *cmd], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):
        if rec_of(rid).get("status") == "running" and rec_of(rid).get("pid"):
            break
        time.sleep(0.1)
    return p


def stop(p, rid):
    """end a background launcher and the command it started"""
    p.kill()
    p.wait()
    pgid = rec_of(rid).get("pgid")
    if pgid:
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except OSError:
            pass


def test_gate():
    """1, 8, 9: the TEST gate is a table row that names the id as a whole token and does not say pending / refused / DONE; a split name `test_clean` is the holdout too"""
    log = ("# TEST log\n\nRule: once per finished design. Pending, NOT approved yet: T.free must not run.\nT.line is fine, see below.\n\n"
           "| date | loop | what was evaluated | who / which run | numbers | note |\n|---|---|---|---|---|---|\n"
           "| 2026-11-10 | 30 | design A | run id T.ok (`evaluate.py a --test-id T.ok`) | - | the person's OK of 2026-11-10 |\n"
           "| 2026-11-10 | 30 | design B | run id T.sent. | - | OK |\n"
           "| 2026-11-10 | 30 | design C | T.pend | - | pending, the person has not answered |\n"
           "| 2026-11-10 | 30 | design D | T.na | - | NOT approved yet |\n"
           "| 2026-11-10 | 30 | design E | T.ref | - | refused by the person |\n"
           "| 2026-11-10 | 30 | design F | T.den, T.wd | - | denied; withdrawn |\n"
           "| 2026-11-10 | 30 | design G | T.two | - | pending |\n| 2026-11-11 | 30 | design G | T.two | - | OK of 2026-11-11 |\n"
           "| 2026-11-10 | 30 | design H | T.done | DONE, once | OK |\n| 2026-11-10 | 30 | design I | T.used | - | already used |\n"
           "| 2026-11-10 | 30 | design J | T.nyr | pending, not yet run | OK (D80) |\n"
           "| 2026-11-10 | 30 | design K | T.base | vs T.cmp, T.base0 | OK |\n"
           "| 2026-11-10 | 30 | design L | T.dec | - | granted by D28 (the person: one TEST check) |\n"
           "| 2026-11-10 | 30 | design M | T.aw | - | awaiting the OK of the person |\n")
    for rid in ("T.ok", "T.sent", "T.two", "T.nyr", "T.base", "T.dec"):
        assert RUN.test_log_opens(log, rid), rid
    for rid in ("T.free", "T.line", "T.pend", "T.na", "T.ref", "T.den", "T.wd", "T.done", "T.used", "T.o", "T.ok2", "T.ok.b", "T", "T.none", "T.cmp", "T.base0", "T.aw"):
        assert not RUN.test_log_opens(log, rid), rid                    # T.cmp is named in another cell (a comparison), not as the run
    assert "NEW id" in RUN.test_log_refusal(log, "T.done") and "NEW id" in RUN.test_log_refusal(log, "T.used"), "a used run is logged again under a new id"
    assert "no OK" in RUN.test_log_refusal(log, "T.pend") and "not approved" in RUN.test_log_refusal(log, "T.na")
    assert "not named in the 'who / which run' cell" in RUN.test_log_refusal(log, "T.free") and "cell" in RUN.test_log_refusal(log, "T.cmp")
    if os.path.exists(LANE_LOG):                                      # the rows of the lane project's real log (read only): a past use says DONE, a fresh row opens
        lane = open(LANE_LOG, encoding="utf-8").read()
        assert RUN.test_log_opens(lane, "E268.p") and not RUN.test_log_opens(lane, "E230.f1p") and not RUN.test_log_opens(lane, "E230.f1"), "the lane log's rows"
    # the launcher: a free-text line opens nothing, a table row does, the same ok id is never started again, --again or not
    set_cfg()
    open(f"{T}/TEST.md", "w").write(log)
    for rid, code in (("T.free", 2), ("T.pend", 2), ("T.done", 2), ("T.ok", 0)):
        r = go(rid, ["bash", "-c", "echo x"], "--no-git", "--split", "test")
        assert r.returncode == code, (rid, r.stdout)
    assert "NEW id" in go("T.done", ["bash", "-c", "echo x"], "--no-git", "--split", "test").stdout
    again = go("T.ok", ["bash", "-c", "echo x"], "--no-git", "--split", "test", "--again")
    assert again.returncode == 2 and "NEW id" in again.stdout and os.path.exists(f"{T}/runs/T.ok/done"), again.stdout
    # a name that starts with the holdout and an underscore is the holdout (lane drivers: test_clean), as the launcher's split and in the command
    for argv in (["--split", "test_clean"], ["--split=test_clean"], ["--split=val,test_x"], ["--split", "Test_Clean"]):
        r = go("T.cmd", ["bash", "-c", "echo x", "driver.py", *argv], "--no-git", "--split", "val")
        assert r.returncode == 2 and "asks for the holdout split" in r.stdout, (argv, r.stdout)
    assert go("T.tc", ["bash", "-c", "echo x"], "--no-git", "--split", "test_clean").returncode == 2, "the launcher's own test_clean needs a row too"
    open(f"{T}/TEST.md", "a").write("| 2026-11-12 | 30 | clean set | T.tc | - | OK |\n")
    assert go("T.tc", ["bash", "-c", "echo x"], "--no-git", "--split", "test_clean").returncode == 0
    for cmdl in (["x", "--split", "tested"], ["x", "--split=testing"], ["x", "--split", "test.json"], ["x", "--split=val"]):
        assert not RUN.names_split(cmdl, "test"), cmdl
    assert RUN.names_split(["x", "--split", "final"], "test", ("final",)) and not RUN.names_split(["x", "--split", "final"], "test")
    set_cfg(holdout_aliases=["confirm"])                              # project.json holdout_aliases
    r = go("T.alias", ["bash", "-c", "echo x --split confirm"], "--no-git", "--split", "val")
    assert r.returncode == 2 and "asks for the holdout split" in r.stdout, r.stdout
    set_cfg()


def test_command_pid():
    """2: the command's pid is in the record right after the start; a killed launcher with a living command is refused (with the pid) and keeps its GPU slot"""
    set_cfg()
    p = background("L.pid", ["bash", "-c", "sleep 120"], "--gpu", "--gpu-slots", "1")
    try:
        rec = rec_of("L.pid")
        pid, pgid = rec.get("pid"), rec.get("pgid")
        assert pid and pgid == pid and rec.get("status") == "running", rec                        # while it runs, not only after the command ended
        assert RUN.command_alive(rec) and RUN.pid_alive(pid)
        p.kill()                                                                                  # the launcher dies, the command (its own process group) lives on
        p.wait()
        time.sleep(1.5)
        assert RUN.pid_alive(pid)
        old = RUN.STALE_S
        RUN.STALE_S = 1.0
        try:
            assert RUN.state_of(f"{T}/runs/L.pid") == "stalled" and RUN.command_alive(rec_of("L.pid"))
            if RUN.proc_stat(pid):                                                                # Linux: the start time tells a reused pid from our command
                assert rec["pid_start"] and not RUN.command_alive(dict(rec_of("L.pid"), pid_start=rec["pid_start"] + 1)), "a reused pid is not our command"
            assert not RUN.command_alive(dict(rec_of("L.pid"), status="failed")) and not RUN.command_alive({})
        finally:
            RUN.STALE_S = old
        again = go("L.pid", ["bash", "-c", "echo x"], "--no-git", e=stale)
        assert again.returncode == 2 and f"its command still runs (pid {pid}" in again.stdout and "attempt after" not in str(rows().get("L.pid", {}).get("notes")), again.stdout
        other = go("L.other", ["bash", "-c", "echo x"], "--no-git", "--gpu", "--gpu-slots", "1", "--wait-gpu", "0", e=stale)
        assert other.returncode == 1 and rows()["L.other"]["status"] == "gpu_busy", "the slot of a run whose command lives stays taken: " + other.stdout
        os.killpg(int(pgid), signal.SIGKILL)                                                      # the command is gone: the old behaviour (stalled, resumed, the slot is free)
        for _ in range(100):
            if not RUN.command_alive(rec_of("L.pid")):
                break
            time.sleep(0.1)
        redo = go("L.pid", ["bash", "-c", "echo x"], "--no-git", "--gpu", "--gpu-slots", "1", e=stale)
        assert redo.returncode == 0 and "attempt after stalled" in rows()["L.pid"]["notes"], redo.stdout
    finally:
        stop(p, "L.pid")


def test_ledger():
    """3: once a row has started its card is frozen (changing and filling), --status is one run.py writes and never back to planned, a verdict needs a rule"""
    nocard = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "L.nocard", "--no-card", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)
    assert nocard.returncode == 0, nocard.stdout
    row = rows()["L.nocard"]
    assert row["started"] and row["rule"] == "" and row["status"] == "ok"
    filled = led("note", "L.nocard", "--rule", "keep if x >= 1")
    assert filled.returncode != 0 and "written before the run" in filled.stdout + filled.stderr and rows()["L.nocard"]["rule"] == "", "a rule filled after the run is a change"
    assert led("note", "L.nocard", "--rule", "keep if x >= 1", "--force").returncode == 0
    assert rows()["L.nocard"]["rule"] == "keep if x >= 1" and "rule before the change: (empty)" in rows()["L.nocard"]["notes"], rows()["L.nocard"]
    for bad in ("bogus", "planned"):
        r = led("note", "L.nocard", "--status", bad)
        assert r.returncode != 0 and rows()["L.nocard"]["status"] == "ok", (bad, r.stdout, r.stderr)
    assert led("note", "L.nocard", "--status", "failed").returncode == 0 and rows()["L.nocard"]["status"] == "failed"
    # a planned row is still free; a verdict without a rule is refused unless it is labelled by hand; the existing flows go on
    assert led("add", "L.plan", "--hyp", "h").returncode == 0 and led("note", "L.plan", "--pred", "p", "--rule", "r").returncode == 0, "a planned row may be written"
    assert led("add", "L.norule", "--hyp", "h").returncode == 0
    v = led("note", "L.norule", "--decision", "KEEP (rule met)")
    assert v.returncode != 0 and "no rule" in v.stdout + v.stderr and rows()["L.norule"]["decision"] == "", (v.stdout, v.stderr)
    assert led("note", "L.norule", "--decision", "KEEP by hand: the person looked at it").returncode == 0 and rows()["L.norule"]["decision"].startswith("KEEP by hand")
    assert led("note", "L.norule", "--decision", "diagnostic only").returncode == 0
    assert led("note", "L.norule", "--rule", "r", "--decision", "TIE: effect +0.0000").returncode == 0, "the rule given in the same call is a rule"
    assert go("L.plan", ["bash", "-c", "echo x"], "--no-git").returncode == 0 and led("note", "L.plan", "--result", "r1", "--decision", "KEEP (effect +0.02)").returncode == 0


def test_twins():
    """4: `X.a` running (or finished) and `X_a` started later name one run folder: refused"""
    p = background("TW.a", ["bash", "-c", "sleep 60"])
    try:
        r = go("TW_a", ["bash", "-c", "echo x"], "--no-git")
        assert r.returncode == 2 and "TW.a is running and names the same run folder" in r.stdout and not os.path.exists(f"{T}/runs/TW_a/run.json"), r.stdout
        assert go("TW.b", ["bash", "-c", "echo x"], "--no-git").returncode == 0, "another folder is no twin"
    finally:
        stop(p, "TW.a")
    assert go("TD.a", ["bash", "-c", "echo x"], "--no-git").returncode == 0
    r = go("TD_a", ["bash", "-c", "echo x"], "--no-git")
    assert r.returncode == 2 and "TD.a finished with exit 0 and names the same run folder" in r.stdout, r.stdout
    assert go("TD_a", ["bash", "-c", "echo x"], "--no-git", "--again").returncode == 0, "--again is a deliberate repeat"


def test_snapshot():
    """5: the scratch scripts a command names stay in a capped snapshot (then the newest); the drift note lists the code folders' files first, with a count"""
    top = tempfile.mkdtemp(prefix="ar_launch_snap_")
    old_cap = RUN.SCRATCH_MAX_FILES
    try:
        os.makedirs(f"{top}/code")
        os.makedirs(f"{top}/temp/sub")
        open(f"{top}/code/a.py", "w").write("1\n")
        open(f"{top}/.gitignore", "w").write("/temp/**\n")
        files = {"temp/zz_named.sh": 1000, "temp/a_oldest.py": 2000, "temp/b_mid.py": 3000, "temp/c_new.py": 4000, "temp/sub/d_newest.py": 5000}      # path -> mtime
        for f, t in files.items():
            open(f"{top}/{f}", "w").write("x = 1\n")
            os.utime(f"{top}/{f}", (t, t))
        for args in (["init", "-q"], ["config", "user.name", "selftest"], ["config", "user.email", "selftest@example.invalid"], ["add", "-A"], ["commit", "-q", "-m", "c0"]):
            subprocess.run(["git", "-c", "safe.directory=*", "-C", top, *args], check=True, capture_output=True)
        cfg = dict(code_dirs=["code"], fingerprint_extra=[], scratch="temp")
        RUN.SCRATCH_MAX_FILES = 3
        assert RUN.scratch_scripts(top, top, cfg) == ["temp/b_mid.py", "temp/c_new.py", "temp/sub/d_newest.py"], "without a command: the newest"
        assert RUN.SCRATCH_SKIPPED[0] == 2
        for cmd in (["bash", "temp/zz_named.sh", "--x"], ["bash", "-c", "cd temp && bash zz_named.sh"], ["bash", f"{top}/temp/zz_named.sh"], ["bash", "--cfg=./temp/zz_named.sh"]):
            assert RUN.scratch_scripts(top, top, cfg, cmd) == ["temp/c_new.py", "temp/sub/d_newest.py", "temp/zz_named.sh"], (cmd, RUN.scratch_scripts(top, top, cfg, cmd))
        _, tag, note = RUN.git_snapshot(top, cfg, "t", "S.cap", "m", ["bash", "temp/zz_named.sh"])
        held = subprocess.run(["git", "-C", top, "ls-tree", "-r", "--name-only", tag], capture_output=True, text=True).stdout.split()
        assert tag == "t/S.cap" and "temp/zz_named.sh" in held and "temp/a_oldest.py" not in held and "2 scratch script(s) not in the tag" in note, (held, note)
    finally:
        RUN.SCRATCH_MAX_FILES = old_cap
        shutil.rmtree(top, ignore_errors=True)
    # the drift note: 12 files of the code folder and the extra file aaa.json (which sorts first) changed while the run went
    set_cfg()
    edit = "echo 1 > aaa.json; for i in 1 2 3 4 5 6 7 8 9 10 11 12; do echo x=$i > code/f$i.py; done; echo ran"
    r = go("L.drift", ["bash", "-c", edit])
    drift = rec_of("L.drift").get("code_drift", {})
    assert r.returncode == 0 and drift.get("n") == 13 and drift.get("n_other") == 1 and "aaa.json" not in drift["paths"] and drift["paths"][0].startswith("code/"), (r.stdout, drift)
    assert "13 files, 1 of them outside the code folders" in rows()["L.drift"]["notes"], rows()["L.drift"]["notes"]
    for f in os.listdir(f"{T}/code"):
        if f != "a.py":
            os.remove(f"{T}/code/{f}")


def test_notes():
    """6: the words that cleared a hold stay in the notes when the resumed run writes its final row"""
    assert go("L.stop", ["bash", "-c", "exit 75"], "--no-git").returncode == 75
    c = subprocess.run([sys.executable, f"{HERE}/hold.py", "clear", "L.stop", "--words", "Mirko: go on"], capture_output=True, text=True, env=env)
    assert c.returncode == 0 and "Mirko: go on" in rows()["L.stop"]["notes"], c.stdout + c.stderr
    assert go("L.stop", ["bash", "-c", "echo again"], "--no-git").returncode == 0
    notes = rows()["L.stop"]["notes"]
    assert "Mirko: go on" in notes and "attempt after stopped" in notes and "ON HOLD" in notes and rows()["L.stop"]["status"] == "ok", notes


def test_card_fields():
    """7: card_id / card_sha of experiment.card_for_run / rule_sha go into the record; no card, an old experiment.py and a crashing one leave them out, the run goes on"""
    stub = ("import sys, types; sys.path.insert(0, sys.argv[1]); mode = sys.argv[2]; m = types.ModuleType('experiment')\n"
            "if mode == 'card': m.card_for_run = lambda rid: {'id': 'E9'}; m.rule_sha = lambda exp: 'abc123'\n"
            "if mode == 'none': m.card_for_run = lambda rid: None; m.rule_sha = lambda exp: 'x'\n"
            "if mode == 'crash': m.card_for_run = lambda rid: 1 / 0\n"
            "sys.modules['experiment'] = m\n"
            "import run; sys.argv = ['run.py'] + sys.argv[3:]; sys.exit(run.main())")
    for mode in ("card", "none", "old", "crash"):
        rid = f"L.card_{mode}"
        r = subprocess.run([sys.executable, "-c", stub, HERE, mode, "--id", rid, "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)
        rec = rec_of(rid)
        assert r.returncode == 0 and rec.get("status") == "ok", (mode, r.stdout, r.stderr[-300:])
        assert (rec.get("card_id"), rec.get("card_sha")) == (("E9", "abc123") if mode == "card" else (None, None)), (mode, rec)


def test_hold_before_start():
    """10: a hold that arrives while the snapshot is made still stops the run before its command starts"""
    stub = ("import sys, os, json; sys.path.insert(0, sys.argv[1]); import run; folder = sys.argv[2]\n"
            "def snap(*a):\n    os.makedirs(folder, exist_ok=True); json.dump({'who': 'queue', 'why': 'arrived during the snapshot'}, open(os.path.join(folder, 'HOLD'), 'w')); return ('abc1234', 't/x', '')\n"
            "run.git_snapshot = snap; sys.argv = ['run.py'] + sys.argv[3:]; sys.exit(run.main())")
    r = subprocess.run([sys.executable, "-c", stub, HERE, f"{T}/runs/L.late", "--id", "L.late", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-drift-check", "--", "bash", "-c", f"echo ran > {T}/ran_late"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "on HOLD" in r.stdout and not os.path.exists(f"{T}/ran_late"), (r.returncode, r.stdout, r.stderr[-300:])
    assert rec_of("L.late")["status"] == "held" and "started" not in rec_of("L.late") and rows()["L.late"]["status"] == "held", rec_of("L.late")


def test_settle():
    """11: the settle wait ends when the slot's run logged its first step (log.jsonl row newer than the slot, train.log `step N:` after the last `=== `), the clock stays the ceiling"""
    assert RUN.train_dir(T, {"trainsuite": {"runs": "rr"}}, "a.b") == os.path.join(T, "rr", "a_b") and RUN.train_dir(T, {"trainsuite": {}}, "a.b") == os.path.join(T, "runs", "a_b")
    assert RUN.train_dir(T, {}, "a.b") == ""
    slots, out = f"{T}/slots", f"{T}/trainrun"
    os.makedirs(f"{slots}/0")
    os.makedirs(out)
    owner = dict(id="loader", pid=os.getpid(), run_dir=f"{T}/runs/loader", out_dir=out, taken=RUN.now(), token="tok")
    json.dump(owner, open(f"{slots}/0/run.json", "w"))
    gs = RUN.GpuSlot(T, 1, "me", f"{T}/runs/me", base=slots)
    loading = lambda: [i for i, _ in gs.settling(120)]       # noqa: E731
    assert loading() == ["loader"] and gs.settling(0) == [], "no log yet: the clock holds; settle 0 = off"
    with open(f"{out}/train.log", "w") as f:
        f.write("=== 2026-11-10 run loader\nloading the data\n")
    assert loading() == ["loader"], "started, but no step yet"
    with open(f"{out}/train.log", "a") as f:
        f.write("step 1: total 3.1 lr 1e-4 grad 2.0\n")
    assert loading() == [], "the first step is logged: loaded"
    # a resumed run: the steps of its earlier attempt lie above the new `=== ` line (and in log.jsonl with an old time)
    os.remove(f"{out}/train.log")
    old = time.time() - 3600
    with open(f"{out}/train.log", "w") as f:
        f.write("=== first attempt\nstep 500: total 1.0\n")
    os.utime(f"{out}/train.log", (old, old))
    with open(f"{out}/log.jsonl", "w") as f:
        f.write(json.dumps({"step": 500, "t": old, "total": 1.0, "gpu_gb": 7.0}) + "\n" + json.dumps({"step": 500, "val": {"total": 1.1}}) + "\n")
    assert loading() == ["loader"], "the old steps of an earlier attempt do not count"
    with open(f"{out}/train.log", "a") as f:
        f.write("=== resumed run loader\n")
    assert loading() == ["loader"]
    with open(f"{out}/log.jsonl", "a") as f:
        f.write(json.dumps({"step": 501, "t": time.time() + 1, "total": 0.9, "gpu_gb": 7.0}) + "\n")
    assert loading() == [], "a log.jsonl row newer than the slot"
    json.dump(dict(owner, out_dir=""), open(f"{slots}/0/run.json", "w"))
    assert loading() == ["loader"], "a slot without a run folder (an older launcher, a project without a trainsuite): the clock only"
    json.dump(dict(owner, taken=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 600))), open(f"{slots}/0/run.json", "w"))
    assert loading() == [], "the clock is the ceiling"
    shutil.rmtree(slots)
    mine = RUN.GpuSlot(T, 1, "mine", f"{T}/runs/mine", base=slots, out_dir=out)
    assert mine.try_take() and json.load(open(f"{slots}/0/run.json"))["out_dir"] == out, "the owner file names the run folder"


TESTS = ("test_gate", "test_command_pid", "test_ledger", "test_twins", "test_snapshot", "test_notes", "test_card_fields", "test_hold_before_start", "test_settle")
try:
    set_cfg()
    for name in sys.argv[1:] or TESTS:                  # one test by name: python selftest_launch.py test_settle
        globals()[name]()
        print(f"  ok {name}")
    print("launcher selftest passed")
finally:
    shutil.rmtree(T, ignore_errors=True)

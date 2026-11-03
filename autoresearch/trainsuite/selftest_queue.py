"""Self-test of the training queue (trainqueue.py): real run.py launchers, a real runner process and a toy trainer in a throw-away git project, one minute.
It checks what the queue promises: order and --after, the runner ends by itself, `hold` stops a running trainer with a checkpoint and keeps the rest from starting,
`resume` continues from that checkpoint on the code of the first attempt (a pin), a run whose launcher was killed is restarted, a run that was started without
the queue is held and resumed from the captured command, a failed dependency blocks, and nothing is done without the person's words."""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from . import trainqueue as TQ
from .cli import main as cli_main

AR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINER = '''"""toy trainer: steps of `sleep` seconds, a checkpoint every 5 steps, honours CONTROL (checkpoint, STOPPED, exit 75), resumes, records every start"""
import json, os, sys, time
sys.path.insert(0, %r)
from trainkit import control
cfg = json.load(open(sys.argv[1]))
rd = os.path.join("runs", cfg["name"])
os.makedirs(rd, exist_ok=True)
control.clear_stale(rd)
ck, step = os.path.join(rd, "ckpt.json"), 0
if os.path.exists(ck):
    step = json.load(open(ck))["step"]
with open(os.path.join(rd, "starts.jsonl"), "a") as f:
    f.write(json.dumps({"from": step, "commit": os.environ.get("AR_COMMIT", ""), "t": time.time()}) + "\\n")
json.dump({"steps": cfg["steps"]}, open(os.path.join(rd, "config.json"), "w"))
while step < cfg["steps"]:
    why = control.stop_requested(rd)
    if why is not None:
        json.dump({"step": step}, open(ck, "w"))
        control.mark_stopped(rd, step, why)
        sys.exit(control.EXIT_STOPPED)
    time.sleep(cfg["sleep"])
    step += 1
    with open(os.path.join(rd, "log.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "total": 1.0}) + "\\n")
    if step %% 5 == 0:
        json.dump({"step": step}, open(ck, "w"))
sys.exit(1 if cfg.get("fail") else 0)
''' % AR


def until(pred, timeout=60.0, every=0.3, what="a condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(every)
    raise AssertionError(f"timeout ({timeout:g} s) waiting for {what}")


def git(proj, *args):
    return subprocess.run(["git", "-C", proj, *args], capture_output=True, text=True, check=True).stdout.strip()


def make_project(tmp):
    proj = os.path.join(tmp, "qproj")
    os.makedirs(os.path.join(proj, "code"))
    os.makedirs(os.path.join(proj, "records", "runs"))
    open(os.path.join(proj, "code", "trainer.py"), "w").write(TRAINER)
    json.dump({"steps": 20, "name": "x", "sleep": 0.05, "fail": False}, open(os.path.join(proj, "base.json"), "w"))
    json.dump({"tag_prefix": "q", "ledger": "LEDGER.csv", "ledger_md": "LEDGER.md", "run_records": "records/runs", "code_dirs": ["code"], "scratch": "",
               "trainsuite": {"out": "composed", "train": "python3 code/trainer.py {config}", "runs": "runs"}}, open(os.path.join(proj, "project.json"), "w"))
    open(os.path.join(proj, ".gitignore"), "w").write("runs/\ncomposed/\nrecords/\n")
    git(proj, "init", "-q")
    git(proj, "config", "user.name", "t")
    git(proj, "config", "user.email", "t@example.com")
    git(proj, "add", "-A")
    git(proj, "commit", "-q", "-m", "toy")
    return proj


def cli(proj, *args):
    """(exit code, stdout, stderr) of `trainsuite queue ...` against the toy project"""
    out, err = io.StringIO(), io.StringIO()
    old = os.environ.get("AR_PROJECT")
    os.environ["AR_PROJECT"] = proj
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(["queue", *args])
    finally:
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
    return rc, out.getvalue(), err.getvalue()


def starts(proj, rid):
    p = os.path.join(proj, "runs", rid.replace(".", "_"), "starts.jsonl")
    return [json.loads(x) for x in open(p).read().splitlines()] if os.path.exists(p) else []


def logged(proj, rid):
    return TQ.last_step(os.path.join(proj, "runs", rid.replace(".", "_")))[0]


def rec(proj, rid):
    p = os.path.join(proj, "records", "runs", rid, "run.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def is_done(proj, rid):
    return os.path.exists(os.path.join(proj, "records", "runs", rid, "done"))


def add(proj, rid, steps, *extra):
    rc, out, err = cli(proj, "add", "base.json", "--id", rid, "--set", f"steps={steps}", "--no-card", *extra)
    assert rc == 0, (rc, out, err)
    return out


def queue_checks(tmp, ok):
    env = {"TS_QUEUE_EVERY": "0.5", "TS_QUEUE_IDLE": "3", "AR_HEARTBEAT_S": "0.5", "AR_STALE_S": "4", "AR_RAM_CHECKS": "1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    proj = make_project(tmp)
    c = TQ.context(proj)
    def item(rid):
        return next(i for i in TQ.load(c)["items"] if i["id"] == rid)

    def kill_launcher(rid):
        lp = rec(proj, rid)["launcher_pid"]
        os.killpg(lp, signal.SIGKILL)                                             # the launcher was started as a session leader; its trainer has a session of its own
        until(lambda: not TQ.proc_argv(lp), 10, what="the launcher is dead")

    def foreign(rid, steps=300, cmd=None):
        """a run.py launcher the queue did not start"""
        fp = os.path.join(proj, "composed", rid + ".json")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        json.dump({"steps": steps, "name": rid, "sleep": 0.05, "fail": False}, open(fp, "w"))
        return subprocess.Popen([sys.executable, os.path.join(TQ.TOOLS, "run.py"), "--id", rid, "--gpu", "--split", "train", "--no-card", "--", *(cmd or ["python3", "code/trainer.py", f"composed/{rid}.json"])],
                                cwd=proj, env=dict(os.environ, AR_PROJECT=proj), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        # 1 order, --after, parallel start, the runner ends by itself
        add(proj, "a", 250, "--no-runner")
        add(proj, "b", 10, "--no-runner", "--after", "a")
        add(proj, "c", 10, "--no-runner")
        assert cli(proj, "add", "base.json", "--id", "a", "--no-card", "--no-runner")[0] == 2, "an id twice in the queue is refused"
        rc, out, _ = cli(proj)
        assert rc == 0 and "queued" in out and "after a" in out, out
        pid = TQ.ensure_runner(c)
        until(lambda: all(is_done(proj, r) for r in "abc"), 90, what="a, b, c done")
        a_end, c_start, b_start = os.path.getmtime(os.path.join(proj, "records", "runs", "a", "done")), starts(proj, "c")[0]["t"], starts(proj, "b")[0]["t"]
        assert c_start < a_end < b_start, (c_start, a_end, b_start, "c (no dependency) starts while a runs, b only after a is done")
        until(lambda: TQ.runner_alive(c) == 0, 30, what="the runner ends by itself when nothing is left")
        assert "nothing left to do" in open(c.runner + ".log").read(), "the runner ended because nothing was left, not because it crashed"
        proj2 = os.path.join(tmp, "qproj2")                                       # a second problem on the same GPU (its slot folder is the first one's): its own list
        os.makedirs(os.path.join(proj2, "records", "runs"))
        json.dump({"run_records": "records/runs", "gpu_slots_dir": os.path.relpath(c.slots, proj2), "trainsuite": {"runs": "runs"}}, open(os.path.join(proj2, "project.json"), "w"))
        keep = os.environ.get("AR_PROJECT")
        c2 = TQ.context(proj2)
        os.environ["AR_PROJECT"] = keep if keep is not None else proj
        assert c2.slots == c.slots and c2.qpath != c.qpath and c2.runner != c.runner, (c.qpath, c2.qpath)
        ok("queue: items start in order, --after waits for the done flag, an item without a dependency starts beside a running one, the runner ends when nothing is left")

        # 2 hold --all stops a running trainer with a checkpoint, nothing new starts; resume continues from the checkpoint on the code of the first attempt
        add(proj, "h1", 300)
        add(proj, "h2", 10, "--after", "h1")
        until(lambda: logged(proj, "h1") >= 12, 60, what="h1 under way")
        assert cli(proj, "hold", "--all")[0] == 2 and cli(proj, "hold", "--all", "--words", "  ")[0] == 2, "hold needs the person's words"
        assert rec(proj, "h1").get("status") == "running", "a refused hold changed nothing"
        rc, out, err = cli(proj, "hold", "--all", "--words", "Mirko: stop it", "--wait", "40")
        step = logged(proj, "h1")
        assert rc == 0 and "h1: stopped at step" in out and "paused" in out, (out, err)
        until(lambda: rec(proj, "h1").get("status") == "stopped", 20, what="the record says stopped")
        assert os.path.exists(os.path.join(proj, "records", "runs", "h1", "HOLD")) and 10 < step < 300, (step, out)
        assert json.load(open(os.path.join(proj, "runs", "h1", "ckpt.json")))["step"] == step, "the checkpoint is the step it stopped at"
        add(proj, "hp", 10)                                                       # an independent item added during the pause: it must not start
        time.sleep(3)
        assert not rec(proj, "h2") and not rec(proj, "hp") and not item("hp").get("launched") and len(starts(proj, "h1")) == 1, "paused: nothing started, h1 was not started again"
        assert "PAUSED" in cli(proj)[1]
        first = starts(proj, "h1")[0]["commit"]
        assert first, "the first attempt ran a snapshot commit"
        with open(os.path.join(proj, "code", "trainer.py"), "a") as f:
            f.write("# a change of the code while h1 is held\n")                  # a new snapshot would be another commit
        assert cli(proj, "resume", "--all")[0] == 2, "resume needs the person's words"
        rc, out, err = cli(proj, "resume", "--all", "--words", "Mirko: go on")
        assert rc == 0 and f"h1: starts again from its last checkpoint (step {step} of 300)" in out and first[:10] in out, (out, err)
        until(lambda: is_done(proj, "h1") and is_done(proj, "h2") and is_done(proj, "hp"), 90, what="h1, h2 and hp done after the resume")
        s = starts(proj, "h1")
        assert len(s) == 2 and s[1]["from"] == step and s[1]["commit"] == first and logged(proj, "h1") == 300, s
        assert any(n.startswith("HOLD.cleared") for n in os.listdir(os.path.join(proj, "records", "runs", "h1"))), "the hold was cleared with the words, kept as a record"
        assert starts(proj, "h2")[0]["t"] > s[1]["t"], "h2 (after h1) started after h1 was resumed and done"
        ok("hold --all: the trainer stops at a step boundary with a checkpoint and a hold, the rest does not start; resume --all: the words clear the hold, the run continues from the "
           "checkpoint (step %d) on the code of its first attempt although the code changed, then the next item starts" % step)

        # 3 resume --fresh-code takes the code of now
        subprocess.run(["git", "-C", proj, "commit", "-q", "-am", "toy 2"], check=True)
        add(proj, "h3", 200)
        until(lambda: logged(proj, "h3") >= 8, 60, what="h3 under way")
        assert cli(proj, "hold", "h3", "--words", "w", "--wait", "40")[0] == 0 and cli(proj, "hold", "nosuch", "--words", "w")[0] == 2
        first3 = starts(proj, "h3")[0]["commit"]
        with open(os.path.join(proj, "code", "trainer.py"), "a") as f:
            f.write("# another change\n")
        assert cli(proj, "resume", "h3", "--words", "w", "--fresh-code")[0] == 0
        until(lambda: is_done(proj, "h3"), 60, what="h3 done")
        s3 = starts(proj, "h3")
        assert len(s3) == 2 and s3[1]["from"] > 0 and s3[1]["commit"] and s3[1]["commit"] != first3, s3
        ok("resume --fresh-code continues from the checkpoint on a new snapshot of the code (the hot-fix loop); hold of a name that is not there is refused")

        # 4 a launcher that was killed: its trainer still runs in a session of its own (an orphan: never started a second time); when the trainer is gone too (WSL shutdown)
        # the runner restarts the run from its checkpoint
        add(proj, "k", 300)
        until(lambda: logged(proj, "k") >= 40, 60, what="k under way (past the launcher's first seconds)")
        kill_launcher("k")
        until(lambda: TQ.state(c, item("k")) == "orphan", 30, what="k is an orphan: the launcher is dead, the trainer is not")
        n0 = logged(proj, "k")
        time.sleep(3)
        assert len(starts(proj, "k")) == 1 and logged(proj, "k") > n0, "the orphan goes on and is not started a second time"
        for pid in TQ.orphans(c, item("k")):
            os.kill(pid, signal.SIGKILL)                                          # now nothing trains
        until(lambda: TQ.state(c, item("k")) == "interrupted", 30, what="k is interrupted: launcher and trainer are gone")
        time.sleep(3)
        assert len(starts(proj, "k")) == 1 and "killed in this boot" in cli(proj)[1], "killed by hand in this boot: the runner leaves it to the person (resume)"
        with TQ.edit(c) as q:                                                     # the machine went down: the launcher died in an earlier boot
            next(i for i in q["items"] if i["id"] == "k")["boot"] = "an-earlier-boot"
        until(lambda: len(starts(proj, "k")) == 2, 60, what="the runner restarted k")
        until(lambda: is_done(proj, "k"), 90, what="k done")
        sk = starts(proj, "k")
        assert sk[1]["from"] >= 10 and logged(proj, "k") == 300 and item("k")["restarts"] == 1 and sk[1]["commit"] == sk[0]["commit"] and sk[0]["commit"], (sk, item("k"))
        ok("a run whose launcher died: its surviving trainer (orphan) is not started a second time; a launcher killed by hand in this boot is left to the person; after a machine "
           "restart (another boot) the runner restarts the run from its last checkpoint on the code of its first attempt")

        # 4b hold stops an orphan too (no launcher to write the hold: the item's own mark keeps it from being restarted); resume continues it
        add(proj, "o", 300)
        until(lambda: logged(proj, "o") >= 40, 60, what="o under way")
        kill_launcher("o")
        until(lambda: TQ.state(c, item("o")) == "orphan", 30, what="o is an orphan")
        rc, out, err = cli(proj, "hold", "o", "--words", "w", "--wait", "40")
        assert rc == 0 and "o: stopped at step" in out, (out, err)
        so = logged(proj, "o")
        time.sleep(3)
        assert TQ.state(c, item("o")) == "held" and len(starts(proj, "o")) == 1 and not TQ.orphans(c, item("o")), "stopped, held, not restarted by the runner"
        assert cli(proj, "resume", "o", "--words", "w")[0] == 0
        until(lambda: is_done(proj, "o"), 90, what="o done")
        assert [x["from"] for x in starts(proj, "o")] == [0, so], starts(proj, "o")
        ok("hold stops an orphaned trainer (checkpoint) and keeps the runner from restarting it; resume continues it from that checkpoint")

        # 5 a run started without the queue is held (command captured from the live launcher) and resumed
        launcher = foreign("d1")
        until(lambda: logged(proj, "d1") >= 8, 60, what="d1 under way")
        rc, out, err = cli(proj, "hold", "d1", "--words", "w", "--wait", "40")
        assert rc == 0 and "d1: stopped at step" in out and launcher.wait(timeout=20) == 75, (out, err)
        step_d = logged(proj, "d1")
        rc, out, err = cli(proj, "resume", "d1", "--words", "w")
        assert rc == 0 and f"step {step_d} of 300" in out, (out, err)
        until(lambda: is_done(proj, "d1"), 90, what="d1 done")
        assert [x["from"] for x in starts(proj, "d1")] == [0, step_d] and logged(proj, "d1") == 300, starts(proj, "d1")
        ok("hold / resume work on a run the queue did not start: the launcher's command line is captured before the stop, the run continues from its checkpoint")

        # 6 a failed dependency blocks; drop; a running item cannot be dropped
        add(proj, "f1", 3, "--set", "fail=true")
        add(proj, "f2", 3, "--after", "f1")
        until(lambda: rec(proj, "f1").get("status") == "failed", 60, what="f1 failed")
        time.sleep(2.5)
        assert not rec(proj, "f2"), "f2 did not start behind a failed f1"
        assert "blocked by f1 (failed)" in cli(proj)[1]
        assert cli(proj, "drop", "f2")[0] == 0 and "f2" not in cli(proj)[1]
        add(proj, "r", 300)
        until(lambda: logged(proj, "r") >= 5, 60, what="r under way")
        assert cli(proj, "drop", "r")[0] == 2, "a running item is held first, not dropped"
        assert cli(proj, "hold", "r", "--words", "w", "--wait", "40")[0] == 0 and cli(proj, "drop", "r")[0] == 0
        ok("a failed dependency blocks its item (shown, never started); drop removes an item, refuses a running one")

        # 7 one item stands in the line for its slot at a time (the order stays): both slots taken by runs the queue did not start, three items queued
        blockers = [foreign("bz1"), foreign("bz2")]
        until(lambda: logged(proj, "bz1") >= 3 and logged(proj, "bz2") >= 3, 60, what="bz1 and bz2 hold both slots")
        for rid in ("w1", "w2", "w3"):
            add(proj, rid, 10)
        until(lambda: rec(proj, "w1").get("status") == "waiting", 30, what="w1 stands in the line")
        time.sleep(5)
        assert [r for r in ("w1", "w2", "w3") if rec(proj, r)] == ["w1"] and not item("w2").get("launched") and not item("w3").get("launched"), [(r, rec(proj, r).get("status")) for r in ("w1", "w2", "w3")]
        # w1 is held while it waits (run.py ends the wait, status held); w2 takes its place in the line
        assert cli(proj, "hold", "w1", "--words", "w")[0] == 0
        until(lambda: rec(proj, "w1").get("status") == "held", 30, what="run.py ended the wait of w1 on the hold")
        until(lambda: rec(proj, "w2").get("status") == "waiting", 30, what="w2 stands in the line now")
        assert cli(proj, "hold", "bz1", "bz2", "--words", "w", "--wait", "40")[0] == 0 and [f.wait(timeout=20) for f in blockers] == [75, 75]
        until(lambda: is_done(proj, "w2") and is_done(proj, "w3"), 90, what="w2 and w3 done once the slots are free")
        assert rec(proj, "w1")["status"] == "held", "w1 stayed held while the others ran"
        rc, out, err = cli(proj, "resume", "w1", "--words", "w")
        assert rc == 0, (out, err)
        until(lambda: is_done(proj, "w1"), 60, what="w1 done after the resume: a run held while it waited starts again")
        assert starts(proj, "w2")[0]["t"] < starts(proj, "w3")[0]["t"], "the line kept its order"
        ok("only the first item stands in the line for a slot while the slots are taken; a hold on it ends its wait and the next one takes the place; resume starts it again")

        # 8 hold --all leaves a run that is not a training alone (an evaluation, a data build: no run folder of its own); its launcher is not captured
        ev = foreign("ev1", cmd=["python3", "-c", "import time; time.sleep(25)"])
        until(lambda: rec(proj, "ev1").get("status") == "running", 30, what="ev1 running")
        rc, out, err = cli(proj, "hold", "--all", "--words", "w", "--wait", "10")
        assert rc == 0 and "ev1: not touched" in out and rec(proj, "ev1").get("status") == "running" and not os.path.exists(os.path.join(proj, "records", "runs", "ev1", "HOLD")), (out, err)
        assert all(i["id"] != "ev1" for i in TQ.load(c)["items"]), "ev1 was not captured into the queue"
        assert cli(proj, "resume", "--all", "--words", "w")[0] == 0
        os.killpg(ev.pid, signal.SIGKILL)
        ok("hold --all does not touch a run that is not a training (no run folder, its command does not name one) and does not put it in the queue")

        # 9 an unreadable list is moved aside and refused: an empty list written over it would lose the holds and the pause
        open(c.qpath, "w").write("{ not json")
        rc, out, err = cli(proj, "hold", "--all", "--words", "w")
        assert rc == 2 and "could not be read" in err and any(".bad." in n for n in os.listdir(c.slots)), (rc, err)
        ok("an unreadable queue file is moved aside (.bad.<time>) and the command refuses: the holds in it are not silently dropped")
    finally:
        for r in os.listdir(os.path.join(proj, "records", "runs")):                    # nothing of the test may stay alive
            rc_ = rec(proj, r)
            for pid in (rc_.get("launcher_pid"),):
                if pid and c.R.pid_alive(pid):
                    try:
                        os.killpg(int(pid), signal.SIGKILL)
                    except OSError:
                        pass
        for name in os.listdir("/proc") if os.path.isdir("/proc") else []:                  # a trainer has a session of its own: a failed test must not leave one
            a_ = TQ.proc_argv(name) if name.isdigit() else None
            if a_ and any(x.endswith("code/trainer.py") for x in a_):
                try:
                    if os.path.realpath(f"/proc/{name}/cwd") == os.path.realpath(proj):
                        os.kill(int(name), signal.SIGKILL)
                except OSError:
                    pass
        rp = TQ.runner_alive(c)
        if rp:
            try:
                os.kill(rp, signal.SIGKILL)
            except OSError:
                pass
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")
    with tempfile.TemporaryDirectory() as tmp:
        queue_checks(tmp, ok)
    print(f"queue selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-test of the job runner on a throw-away problem folder (its own project.json, ledger, run.py records; never the real machine lock, ledger or GPU slots):
the catalogue, the check of a job (keys, arguments against the method, data references, card rules, the gates of run.py a job could slip past), the run.py command a job
becomes, submit / start / idempotence / restart / failure with a fake launcher, the states read from run.py's records, and the whole way with the REAL run.py and the child
process (`python -m jobrunner call`). `python -m jobrunner selftest`; ~40 s."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)
if AR not in sys.path:
    sys.path.insert(0, AR)

from jobrunner import JobError, JobRunner                      # noqa: E402
from jobrunner import catalogue as C                           # noqa: E402
from jobrunner.api import _tool                                # noqa: E402

TOY = '''
import os, json, time


class Toy:
    METHODS = ("work", "gpu", "slow", "notok", "boom", "fire")

    def __init__(self, project):
        self.project = project

    def work(self, n, name="x", flag=False, items=(), src=""):
        """double n."""
        print("noise on stdout")
        return {"ok": True, "n": n * 2, "name": name, "flag": flag, "items": list(items)}

    def gpu(self, n, name="x", flag=False, items=(), src=""):
        """work, as a call that may feed a verdict and takes the GPU."""
        return self.work(n, name, flag, items, src)

    def slow(self, seconds=1.0):
        """sleep."""
        time.sleep(seconds)
        return {"ok": True, "slept": seconds}

    def notok(self):
        """a result that says ok=False."""
        return {"ok": False, "why": "toy"}

    def boom(self):
        """an error."""
        raise ValueError("boom")

    def fire(self, tag):
        """starts a run of its own (a record with the done flag) and returns, like Training.run."""
        d = os.path.join(self.project, "runs", tag)
        os.makedirs(d, exist_ok=True)
        json.dump({"status": "ok"}, open(os.path.join(d, "run.json"), "w"))
        open(os.path.join(d, "done"), "w").close()
        return {"ok": True, "tag": tag}
'''

TOY2 = '''
import os
if not os.environ.get("JR_ALLOW"):
    raise ImportError("needs the other interpreter")


class Toy2:
    def __init__(self, project):
        self.project = project

    def go(self, a, b=1):
        """a suite that can only be imported by another interpreter (torch, say)."""
        return {"ok": True, "a": a, "b": b}
'''

TOY3 = '''
JOBS = {"Toy3.x": dict(suite="toysuite3", cls="Toy3", method="x", how="process", writes=["x3:{a}"])}


class Toy3:
    def __init__(self, project):
        pass

    def x(self, a):
        """a suite that brings its own table."""
        return {"ok": True}
'''

TOY4 = "raise SystemExit('no project here')\n"

TOYS = {
    "Toy.work": dict(suite="toysuite", cls="Toy", method="work", how="process", reads=["path:{src}"], writes=["out:{name}"]),
    "Toy.slow": dict(suite="toysuite", cls="Toy", method="slow", how="process", writes=["slept:{id}"]),
    "Toy.notok": dict(suite="toysuite", cls="Toy", method="notok", how="process"),
    "Toy.boom": dict(suite="toysuite", cls="Toy", method="boom", how="process"),
    "Toy.gpu": dict(suite="toysuite", cls="Toy", method="gpu", how="process", may_feed_verdict=True, needs=dict(gpu=True, gpu_max_used_mb=3000, ram_gb=2, cpu=1)),
    "Toy.fire": dict(suite="toysuite", cls="Toy", method="fire", how="launch", id_from="tag"),
}
N = 0
DEAD = 2 ** 22 + 12345                                              # a pid no process has (above pid_max)


def ok(cond, msg=""):
    global N
    N += 1
    assert cond, msg


def raises(fn, text):
    try:
        fn()
    except JobError as e:
        ok(text in str(e), f"expected {text!r} in {e}")
        return
    raise AssertionError(f"no JobError containing {text!r}")


def project(T):
    os.makedirs(f"{T}/code")
    open(f"{T}/code/a.py", "w").write("x = 1\n")
    for name, text in (("toysuite", TOY), ("toysuite2", TOY2), ("toysuite3", TOY3), ("toysuite4", TOY4)):
        open(f"{T}/{name}.py", "w").write(text)
    open(f"{T}/py", "w").write(f'#!/bin/bash\nexec env JR_ALLOW=1 {sys.executable} "$@"\n')
    os.chmod(f"{T}/py", 0o755)
    TOYS["Toy2.go"] = dict(suite="toysuite2", cls="Toy2", method="go", how="process", python=f"{T}/py")
    open(f"{T}/TEST.md", "w").write("| none |\n")
    json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", runs="runs", schedule="schedule.csv", test_log="TEST.md", code_dirs=["code"],
                   scratch="temp", live_code=["code"], jobrunner=dict(catalogue=TOYS, suites=["toysuite3", "toysuite4"])), open(f"{T}/project.json", "w"))
    G = ["git", "-c", "safe.directory=*", "-C", T]
    subprocess.run(G[:3] + ["init", "-q", T], check=True)
    for k, v in (("user.name", "selftest"), ("user.email", "selftest@example.invalid"), ("core.autocrlf", "false")):
        subprocess.run(G + ["config", k, v], check=True)


def record(T, rid, status, age=0.0, launcher=DEAD, **more):
    """a run record as run.py writes it (created, status, launcher_pid, exit, notes), with a heartbeat that is `age` seconds old"""
    d = f"{T}/runs/{rid}"
    os.makedirs(d, exist_ok=True)
    rec = dict(id=rid, created="2026-11-06T02:54:08", status=status, launcher_pid=str(launcher), **more)
    json.dump(rec, open(f"{d}/run.json", "w"))
    if status == "ok":
        open(f"{d}/done", "w").close()
    hb = f"{d}/heartbeat"
    open(hb, "w").close()
    if age:
        os.utime(hb, (time.time() - age, time.time() - age))


def waitfor(jr, jid, states, secs=90):
    t0 = time.time()
    while time.time() - t0 < secs:
        st = jr._state(jid)[0]
        if st in states:
            return st
        time.sleep(0.5)
    raise AssertionError(f"job {jid} is {jr._state(jid)} after {secs} s")


def job(jid, **kw):
    return dict(id=jid, call="Toy.work", args={"n": 1, "name": "n_" + jid}, **kw)


def test_catalogue(T):
    jr = JobRunner(T)
    names = {r["call"] for r in jr.catalogue()}
    ok({"Training.run", "Evaluation.run", "Evaluation.score", "Runner.predict", "Toy.work", "Toy.fire"} <= names, names)
    ok("Toy3.x" in names, "a suite named in project.json jobrunner.suites brings its own table (JOBS); a suite that exits on import does not stop the catalogue")
    ok(not any(C.check_entry(k, e) for k, e in C.BUILTIN.items()), "the built-in entries are well formed")
    ok(C.check_entry("Toy.work", dict(TOYS["Toy.work"], speed=1)) and C.check_entry("Toy.work", dict(TOYS["Toy.work"], how="fork")) and C.check_entry("X.y", TOYS["Toy.work"]), "bad entries are named")
    row = [r for r in jr.catalogue() if r["call"] == "Toy.work"][0]
    ok(row["parameters"] == "n, name=, flag=, items=, src=", row["parameters"])
    bad = json.load(open(f"{T}/project.json"))
    bad["jobrunner"] = {"catalogue": {"Toy.work": {"how": "fork"}}}
    d2 = tempfile.mkdtemp(prefix="jobrunner_bad_")
    try:
        json.dump(bad, open(f"{d2}/project.json", "w"))
        raises(lambda: JobRunner(d2).catalogue(), "the catalogue is wrong")
    finally:
        shutil.rmtree(d2)
    ok(C.fill("run:{network}", dict(network="a")) == "run:a" and C.fill("run:{job[id]}", dict(job={"id": "x"})) == "run:x" and C.fill("run:{nope}", {}) is None and C.fill("{job[id]}", dict(job="p")) is None)
    ok(C.check_needs(dict(gpu=True, gpu_max_used_mb=3000, ram_gb=2.5, cpu=2)) == [] and len(C.check_needs(dict(ram_gb="6", gpu="yes", cpu=1.5, est_hours=True))) == 4)


def test_check(T):
    jr = JobRunner(T)
    r = jr.check({"id": "a1", "call": "Toy.work", "args": {"n": 3}})
    ok(r["ok"] and r["how"] == "process" and r["state"] == "new" and r["diagnostic"], r)
    cmd = r["command"]
    ok("--no-card" in cmd and "--no-git" in cmd and "--retries" not in cmd and "--gpu" not in cmd and "--after" not in cmd, cmd)
    ok(any(c.startswith("--hyp=Toy.work {") and '"n": 3' in c for c in cmd), "the ledger hypothesis of a diagnostic says what was called, with which arguments")
    tail = cmd[cmd.index("--") + 1:]
    ok(tail[1:4] == ["-m", "jobrunner", "call"] and tail[tail.index("--suite") + 1] == "toysuite" and tail[tail.index("--method") + 1] == "work", tail)
    ok(tail[tail.index("--args-file") + 1].endswith("/.jobs/a1/args.json") and tail[tail.index("--out") + 1].endswith("/.jobs/a1/result.json"))
    ok(any("reads 'path:{src}' could not be filled" in w for w in r["warnings"]), "a reference that the arguments cannot fill is a warning, not a problem")
    probs = lambda j: jr.check(j)["problems"]                              # noqa: E731
    ok(any("unknown key 'speed'" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "speed": 1})))
    ok(any("not in the catalogue" in p for p in probs({"id": "a", "call": "Toy.nothing"})))
    ok(any("'n' is required" in p for p in probs({"id": "a", "call": "Toy.work", "args": {}})))
    ok(any("'m' is not a parameter" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1, "m": 2}})))
    ok(any("'flag' must be true or false" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1, "flag": "yes"}})))
    ok(any("'items' must be a list" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1, "items": "ab"}})))
    ok(any("not a plain name" in p for p in probs({"id": "a b/c", "call": "Toy.work", "args": {"n": 1}})))
    ok(any("not a plain name" in p for p in probs({"call": "Toy.work", "args": {"n": 1}})), "an id is required")
    ok(any("needs must be" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "needs": {"gpus": 2}})))
    ok(any("needs.ram_gb must be a number" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "needs": {"ram_gb": "6"}})), "a text for a number would reach run.py as a huge number")
    ok(any("card must be" in p for p in probs({"id": "a", "call": "Toy.gpu", "args": {"n": 1}, "card": {"hyp": "h"}})))
    ok(any("never feeds a verdict" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "card": {"hyp": "h", "pred": "p", "rule": "r"}})))
    ok(any("card and no_card" in p for p in probs({"id": "a", "call": "Toy.gpu", "args": {"n": 1}, "no_card": True, "card": {"hyp": "h", "pred": "p", "rule": "r"}})))
    ok(any("retries must be" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "retries": -1})) and any("retries must be" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "retries": "2"})))
    ok(any("after names the job itself" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "after": ["a"]})))
    ok(any("not a reference" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "reads": ["t23.C_s1"]})))
    ok(any("reads path:nope.txt: not found" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1, "src": "nope.txt"}})))
    open(f"{T}/there.txt", "w").write("x")
    ok(jr.check({"id": "a", "call": "Toy.work", "args": {"n": 1, "src": "there.txt"}})["reads"] == ["path:there.txt"])
    ok(any("no such run" in p for p in probs({"id": "a", "call": "Toy.work", "args": {"n": 1}, "reads": ["run:ghost"]})))
    ok(jr.check({"id": "a", "call": "Toy.work", "args": {"n": 1}, "_comment": "keys with _ are comments"})["ok"])
    ok(jr.check({"id": "a", "call": "Toy.work", "args": {"n": 1}, "note": "-fix"})["command"].count("--hyp=-fix") == 1, "a text that starts with - is passed as --hyp=...")
    g = jr.check({"id": "g1", "call": "Toy.gpu", "args": {"n": 1}, "card": {"hyp": "h", "pred": "p", "rule": "r"}, "split": "val", "after": ["x1"], "timeout": 30, "retries": 1})["command"]
    ok("--gpu" in g and g[g.index("--gpu-max-mb") + 1] == "3000" and g[g.index("--ram-mb") + 1] == "2048" and "--hyp=h" in g and "--pred=p" in g and "--rule=r" in g and "--require-card" in g
       and "--no-card" not in g and "--no-git" not in g and g[g.index("--split") + 1] == "val" and g[g.index("--after") + 1] == "x1" and g[g.index("--timeout") + 1] == "30"
       and g[g.index("--retries") + 1] == "1", g)
    t2 = jr.check({"id": "t2a", "call": "Toy2.go", "args": {"a": 1, "c": 2}})
    ok(any("'c' is not a parameter" in p for p in t2["problems"]) and not t2["warnings"], "a suite this interpreter cannot import is read by the interpreter of the job: " + str(t2))
    ok(jr.check({"id": "t2b", "call": "Toy2.go", "args": {"a": 1}})["ok"] and [r for r in jr.catalogue() if r["call"] == "Toy2.go"][0]["parameters"] == "a, b=")
    ok(jr.check({"id": "t2c", "call": "Toy2.go", "args": {"a": 1}, "python": "/nonexistent/python"})["warnings"][0].startswith("the arguments were not checked"), "an interpreter that cannot run is a warning")
    legacy = jr.check({"id": "tr1", "config": "training/x.json", "preset": ["fast"]})
    ok(legacy["call"] == "Training.run" and legacy["how"] == "launch" and any("training/x.json: not found" in p for p in legacy["problems"]) and legacy["diagnostic"] is None, legacy)
    ok(any("not a parameter" in p for p in probs({"id": "tr2", "call": "Training.run", "args": {"jobs": {}}})), "the signature of a real suite class is read")
    launch = probs({"id": "f1", "call": "Toy.fire", "args": {"tag": "f1"}, "after": ["x"], "timeout": 5, "split": "val", "card": {"hyp": "h", "pred": "p", "rule": "r"}})
    ok(all(any(k + " has no effect on a `launch` call" in p for p in launch) for k in ("after", "timeout", "split", "card")), launch)
    ok(jr.check({"id": "f2", "call": "Toy.fire", "args": {"tag": "f2"}})["ok"] and jr.check({"call": "Toy.fire", "args": {"tag": "f3"}})["id"] == "f3", "id_from takes the id of the arguments")


def test_gates(T):
    """what run.py would refuse and a job must not slip past"""
    jr = JobRunner(T)
    env = dict(os.environ, AR_PROJECT=T)
    probs = lambda j: jr.check(j)["problems"]                              # noqa: E731
    card = {"hyp": "h", "pred": "x >= 1", "rule": "keep if x >= 1"}
    ok(any("name the holdout split" in p for p in probs({"id": "h1", "call": "Toy.gpu", "args": {"n": 1, "name": "large.test"}, "card": card})), "the TEST split inside the arguments is not visible to run.py's --split gate")
    ok(any("name the holdout split" in p for p in probs({"id": "h1", "call": "Toy.gpu", "args": {"n": 1, "items": ["val", "test_clean"]}, "card": card})))
    ok(not any("holdout" in p for p in probs({"id": "h1", "call": "Toy.gpu", "args": {"n": 1, "name": "latest"}, "card": card})), "a name that merely contains the letters is no hit")
    p = probs({"id": "h2", "call": "Toy.gpu", "args": {"n": 1, "name": "large.test"}, "card": card, "split": "test"})
    ok(any("refused by run.py" in x for x in p) and not any("name the holdout split" in x for x in p), "with split=test the TEST log of run.py decides: " + str(p))
    subprocess.run([sys.executable, f"{AR}/tools/ledger.py", "add", "pl1", "--pred", "x >= 1", "--rule", "keep if x >= 1"], check=True, env=env, capture_output=True)
    ok(any("ledger has a row with a card" in x for x in probs(job("pl1"))), "a planned row with a card cannot be run as a diagnostic")
    ok(not any("ledger has a row" in x for x in probs({"id": "pl1", "call": "Toy.gpu", "args": {"n": 1}, "card": card})))
    H = _tool("run").HOLD
    os.makedirs(f"{T}/runs/hd", exist_ok=True)
    H.write(f"{T}/runs/hd", "selftest", "held for the test")
    ok(any("HOLD marker" in x for x in probs(job("hd"))), "a HOLD marker is found before run.py refuses")
    record(T, "tw.a", "ok")
    ok(any("same run folder" in x for x in probs(job("tw_a"))), "a twin run folder (tw.a / tw_a) is found before run.py refuses")
    record(T, "adopt1", "ok")
    ok(any("did not make" in x for x in probs(job("adopt1"))), "an id that names a run of someone else is not adopted")
    os.makedirs(f"{T}/runs/.jobs/Casey", exist_ok=True)
    json.dump({"spec": {"call": "Toy.work", "args": {}}}, open(f"{T}/runs/.jobs/Casey/job.json", "w"))
    ok(any("only in case" in x for x in probs(job("casey"))), "two ids that differ only in case share a folder on /mnt/c")


def test_submit(T):
    calls = []
    live = subprocess.Popen(["sleep", "60"])                              # a launcher that lives
    dead = subprocess.Popen(["true"])
    dead.wait()
    pids = {"live": live.pid, "dead": dead.pid}
    mode = ["live"]

    def fake(argv, cwd, env, log):
        calls.append((argv, cwd, env, log))
        return pids[mode[0]]
    jr = JobRunner(T, launcher=fake)
    try:
        j = {"id": "s1", "call": "Toy.work", "args": {"n": 4, "name": "s1n", "items": ("a", "b")}}
        r = jr.submit(j)
        ok(r["started"] and r["state"] == "queued" and len(calls) == 1, r)
        ok(calls[0][1] == jr.root and calls[0][2]["AR_PROJECT"] == jr.root and AR in calls[0][2]["PYTHONPATH"].split(os.pathsep), "the child finds the problem and the suites")
        ok(json.load(open(f"{T}/runs/.jobs/s1/args.json")) == {"n": 4, "name": "s1n", "items": ["a", "b"]} and os.path.exists(f"{T}/runs/.jobs/s1/job.json"))
        r2 = jr.submit(j)
        ok(not r2["started"] and r2["state"] == "queued" and len(calls) == 1, "the same job (tuples in the arguments included) is not started twice")
        raises(lambda: jr.submit({"id": "s1", "call": "Toy.work", "args": {"n": 5}}), "another call or other arguments")
        raises(lambda: jr.submit({"id": "s2", "call": "Toy.work", "args": {"n": 1, "name": "s1n"}}), "job s1 is writing that now")
        b = jr.submit({"id": "s3", "call": "Toy.work", "args": {"n": 1, "name": "s3n"}, "reads": ["run:s1"]})
        ok("--after" in calls[-1][0] and calls[-1][0][calls[-1][0].index("--after") + 1] == "s1" and b["state"] == "queued", "reading a run that is a job in progress makes the job wait for it")
        jr.submit({"id": "s6", "call": "Toy.work", "args": {"n": 1, "name": "s6n"}, "reads": ["out:s1n"]})
        ok(calls[-1][0][calls[-1][0].index("--after") + 1] == "s1", "a reference that a live job writes makes the job wait for it, whatever its kind")
        jr.submit({"id": "s7", "call": "Toy.work", "args": {"n": 1, "name": "s7n"}, "reads": ["run:runs/s1/"]})
        ok(calls[-1][0][calls[-1][0].index("--after") + 1] == "s1", "a run may be named by its folder")
        jr.submit(job("fo.a"))
        jr.submit(dict(job("fo.b"), reads=["run:fo_a"]))
        ok(calls[-1][0][calls[-1][0].index("--after") + 1] == "fo.a", "a run may be named by the folder it writes into (fo_a for fo.a): else the job would start on a half-written run")
        p = jr.submit({"id": "s4", "call": "Toy.work", "args": {"n": 1, "name": "s4n"}}, start=False)
        ok(p["state"] == "planned" and jr.status("s4")["state"] == "planned", p)
        n = len(calls)
        jr.start("s4")
        ok(len(calls) == n + 1 and jr._state("s4")[0] == "queued")
        raises(lambda: jr.start("s4"), "queued: not started again")
        mode[0] = "dead"
        jr.submit(job("s5"))
        ok(jr._state("s5")[0] == "failed" and "launcher ended" in jr._state("s5")[1])
        mode[0] = "live"
        n = len(calls)
        jr.submit(job("s5"))
        ok(len(calls) == n + 1 and jr._state("s5")[0] == "queued", "a failed job is started again")
        # the lock: one submit of an id at a time
        os.makedirs(f"{T}/runs/.jobs/lk", exist_ok=True)
        open(f"{T}/runs/.jobs/lk/submit.lock", "w").write(str(live.pid))
        raises(lambda: jr.submit(job("lk")), "another submit or start of this id is running")
        open(f"{T}/runs/.jobs/lk/submit.lock", "w").write(str(dead.pid))
        ok(jr.submit(job("lk"))["started"], "the lock of a dead owner is taken over")
        ok(not os.path.exists(f"{T}/runs/.jobs/lk/submit.lock"), "the lock is released")
        # helper files do not collide with ids
        jr.submit(job("s1.args")), jr.submit(job("s1.result"))
        ok({"s1", "s1.args", "s1.result"} <= {r["id"] for r in jr.list()} and jr.status("s1.args")["id"] == "s1.args", "an id that looks like a helper file name is a job of its own")
        # states from the records of run.py
        for rid, status, age, want in (("r_wait", "waiting", 0, "queued"), ("r_run", "running", 0, "running"), ("r_ok", "ok", 0, "done"), ("r_fail", "failed", 0, "failed"),
                                       ("r_held", "held", 0, "held"), ("r_gpu", "gpu_busy", 0, "failed"), ("r_stale", "running", 99999, "stalled"), ("r_block", "blocked", 0, "blocked"),
                                       ("r_stop", "stopped", 0, "stopped"), ("r_time", "timeout", 0, "failed")):
            jr.submit(dict(job(rid), call="Toy.notok", args={}), start=False)
            record(T, rid, status, age, exit="2", notes="because of the notes")
            ok(jr._state(rid)[0] == want, (rid, jr._state(rid)))
        ok(jr._state("r_fail")[1].endswith("because of the notes") and jr.status("r_fail")["exit_code"] == "2", "the reason and the exit code are read from the fields run.py writes")
        ok(jr.submit(dict(job("r_ok"), call="Toy.notok", args={}))["note"] == "already submitted (done)")
        ok(any(row["id"] == "s1" and row["state"] == "queued" for row in jr.list()) and {r["id"] for r in jr.list("held")} == {"r_held"})
        ok(jr.check({"id": "x9", "call": "Toy.work", "args": {"n": 1}, "reads": ["run:r_fail"]})["problems"][0].endswith("that run is failed"))
        ok(jr.check({"id": "x9", "call": "Toy.work", "args": {"n": 1}, "reads": ["run:r_ok"]})["ok"] and jr.check({"id": "x9", "call": "Toy.work", "args": {"n": 1}, "reads": ["run:r_ok"]})["after"] == [])
        # restart: failed, blocked, stopped and a dead stalled job are started again, a stalled one whose launcher lives is not
        for rid in ("r_fail", "r_block", "r_stop", "r_stale"):
            n = len(calls)
            r = jr.submit(dict(job(rid), call="Toy.notok", args={}))
            ok(r["started"] and len(calls) == n + 1, (rid, r))
            ok(jr._state(rid)[0] == "queued", (rid, jr._state(rid), "a restarted job shows the new attempt, not the old failure"))
        jr.submit(dict(job("r_alive"), call="Toy.notok", args={}), start=False)
        record(T, "r_alive", "running", 99999, launcher=live.pid)
        ok(jr._state("r_alive")[0] == "stalled" and "process lives" in jr._state("r_alive")[1])
        raises(lambda: jr.start("r_alive"), "not started again")
        # a finished job: a record that cannot be read for a moment does not turn it into a failure, and its result is kept
        jr.submit(dict(job("c1"), call="Toy.notok", args={}))
        record(T, "c1", "ok")
        json.dump({"ok": True}, open(f"{T}/runs/.jobs/c1/result.json", "w"))
        open(f"{T}/runs/c1/run.json", "w").write("{")
        ok(jr._state("c1")[0] == "done" and jr.submit(dict(job("c1"), call="Toy.notok", args={}))["note"] == "already submitted (done)" and os.path.exists(f"{T}/runs/.jobs/c1/result.json"))
        # a restarted job keeps the earlier result
        mode[0] = "dead"
        jr.submit(job("rr"))
        mode[0] = "live"
        json.dump({"error": "old"}, open(f"{T}/runs/.jobs/rr/result.json", "w"))
        jr.submit(job("rr"))
        ok(json.load(open(f"{T}/runs/.jobs/rr/result.1.json")) == {"error": "old"} and not os.path.exists(f"{T}/runs/.jobs/rr/result.json"), "the result of an earlier attempt is renamed, not deleted")
        # only jobs in progress conflict, a finished one does not
        jr.submit(dict(job("f1"), writes=["out:same"]), start=False)
        record(T, "f1", "ok")
        ok(jr.check(dict(job("f2"), writes=["out:same"]))["ok"], "the outputs of a finished job do not block a new one")
        raises(lambda: jr.status("nope"), "no job 'nope'")
        raises(lambda: jr.status("../x"), "no job")
        # a launch that fails leaves a failed job, not a planned one that blocks others
        mode[0] = "live"
        bad = JobRunner(T, launcher=lambda *a: (_ for _ in ()).throw(OSError("no such file")))
        raises(lambda: bad.submit(dict(job("lf"), writes=["out:lf"])), "could not be started: OSError")
        ok(bad._state("lf")[0] == "failed" and "no such file" in bad._state("lf")[1] and bad.check(dict(job("lf2"), writes=["out:lf"]))["ok"])
    finally:
        live.kill()
        live.wait()


def test_real(T):
    """the whole way: run.py in the background, the child process, the records, the chain by `reads`, a failure, a result that says ok=False, the launch form, the command line"""
    os.environ.update(AR_MACHINE_LOCK=f"{T}/state/machine_lock.json", AR_SETTLE_S="0")
    os.environ.pop("AR_LOCK_OWNER", None)
    jr = JobRunner(T)
    r = jr.submit({"id": "e.a", "call": "Toy.work", "args": {"n": 21, "name": "ea", "flag": True, "items": [1, "b"]}, "note": "selftest"})
    ok(r["started"], r)
    waitfor(jr, "e.a", ("done", "failed"))
    st = jr.status("e.a")
    ok(st["state"] == "done" and st["result"] == {"ok": True, "n": 42, "name": "ea", "flag": True, "items": [1, "b"]} and st["exit_code"] in (0, "0"), st)
    ok(any("noise on stdout" in line for line in st["log"]), "the method's prints go to the log (stderr), not into the result")
    rows = {line.split(",")[0]: line for line in open(f"{T}/LEDGER.csv", encoding="utf-8").read().splitlines()[1:]}
    ok("e.a" in rows and "no card" in rows["e.a"], "run.py wrote the ledger row and called it a diagnostic")
    jr.submit({"id": "e.slow", "call": "Toy.slow", "args": {"seconds": 3}})
    ok(jr.check({"id": "e.b", "call": "Toy.work", "args": {"n": 1, "name": "eb"}, "reads": ["run:e.slow"]})["after"] == ["e.slow"])
    jr.submit({"id": "e.b", "call": "Toy.work", "args": {"n": 1, "name": "eb"}, "reads": ["run:e.slow"]})
    waitfor(jr, "e.b", ("done", "failed"), 120)
    ok(jr._state("e.slow")[0] == "done" and jr._state("e.b")[0] == "done", "the chain ran in order")
    jr.submit({"id": "e.boom", "call": "Toy.boom", "args": {}})
    ok(waitfor(jr, "e.boom", ("done", "failed")) == "failed" and "ValueError: boom" in json.dumps(jr.status("e.boom")["result"]), jr.status("e.boom"))
    jr.submit({"id": "e.boom", "call": "Toy.boom", "args": {}})                                    # started anew: run.py runs a failed id again
    ok(waitfor(jr, "e.boom", ("done", "failed")) == "failed" and os.path.exists(f"{T}/runs/.jobs/e.boom/result.1.json"), "the second attempt ran and the first result is kept")
    jr.submit({"id": "e.notok", "call": "Toy.notok", "args": {}})
    ok(waitfor(jr, "e.notok", ("done", "failed")) == "failed" and jr.status("e.notok")["result"] == {"ok": False, "why": "toy"}, "ok=False in the result is a failed job")
    f = jr.submit({"id": "e.fire", "call": "Toy.fire", "args": {"tag": "e.fire"}})
    ok(f["started"] and f["state"] == "done" and jr.status("e.fire")["result"] == {"ok": True, "tag": "e.fire"}, f)
    r = jr.run({"id": "e.run", "call": "Toy.work", "args": {"n": 2, "name": "erun"}}, wait=90)
    ok(r["state"] == "done" and r["ok"] is True)
    ok(jr.run({"id": "e.run2", "call": "Toy.boom", "args": {}}, wait=90)["ok"] is False, "run() says ok=False for a job that failed")
    env = dict(os.environ, AR_PROJECT=T)
    led = subprocess.run([sys.executable, f"{AR}/tools/ledger.py", "tail"], capture_output=True, text=True, env=env)
    ok(led.returncode == 0, "the other tools read run_records with the .jobs folder in it: " + led.stderr)
    cli = subprocess.run([sys.executable, "-m", "jobrunner", "JobRunner.list", "--json", "--project", T], capture_output=True, text=True, cwd=AR)
    ok(cli.returncode == 0 and {"e.a", "e.b", "e.boom"} <= {r["id"] for r in json.loads(cli.stdout)}, cli.stdout + cli.stderr)
    json.dump({"id": "e.cli", "call": "Toy.boom", "args": {}}, open(f"{T}/cli_job.json", "w"))
    run_cli = subprocess.run([sys.executable, "-m", "jobrunner", "JobRunner.run", f"{T}/cli_job.json", "--wait", "90", "--project", T], capture_output=True, text=True, cwd=AR)
    ok(run_cli.returncode == 1, "the command `run` exits 1 for a job that failed: " + run_cli.stdout[-200:] + run_cli.stderr[-200:])
    bad = subprocess.run([sys.executable, "-m", "jobrunner", "JobRunner.status", "nope", "--project", T], capture_output=True, text=True, cwd=AR)
    ok(bad.returncode == 2 and "no job 'nope'" in bad.stderr and "Traceback" not in bad.stderr, bad.stderr)
    chk = subprocess.run([sys.executable, "-m", "jobrunner", "JobRunner.check", f"{T}/nojob.json", "--project", T], capture_output=True, text=True, cwd=AR)
    ok(chk.returncode == 2 and "cannot read the job file" in chk.stderr, chk.stderr)
    m = subprocess.run([sys.executable, "-m", "jobrunner", "methods"], capture_output=True, text=True, cwd=AR)
    ok(m.returncode == 0 and "submit  (JobRunner.submit(job, start=True))" in m.stdout, m.stdout)


def main():
    T = tempfile.mkdtemp(prefix="jobrunner_selftest_")
    keep = os.environ.get("JOBRUNNER_KEEP")
    try:
        project(T)
        for fn in (test_catalogue, test_check, test_gates, test_submit, test_real):
            fn(T)
            print(f"  {fn.__name__[5:]:10s} ok")
    finally:
        if not keep:
            shutil.rmtree(T, ignore_errors=True)
    print(f"jobrunner selftest passed ({N} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

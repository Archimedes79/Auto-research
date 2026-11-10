"""Self-test of the interface (api.py): jobs, check, run, status, stop, resume, list, compare, plot and the command line against a toy project with real run.py launchers and a toy trainer
(selftest_queue.py builds it), about two minutes. It checks what the interface promises: a job is validated (a typo shows), `check` starts nothing, `run` starts and `status` reads it, `stop`
puts the id on hold and `resume` needs the person's words and continues from the checkpoint, the command line says the same as the functions (--json prints what they return), an error is a
message with exit 2, the older words keep their meaning. The queue is not part of the class: selftest_queue_api.py tests it as a mixin."""
import contextlib
import io
import json
import os
import tempfile
import threading
import time

import suitecli

from . import api
from . import trainqueue as TQ
from .api import LEGACY
from .cli import main as cli_main
from .selftest_queue import make_project, until


def cli(*args):
    """(exit code, stdout, stderr) of `python -m trainsuite ...`"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(list(args))
    return rc, out.getvalue(), err.getvalue()


def raises(fn, text, *a, **kw):
    try:
        fn(*a, **kw)
    except api.TrainingError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected a TrainingError containing {text!r}")


def toy(tmp):
    """a toy project (real run.py launchers, a toy trainer that logs a time and a gradient norm) with the environment of the self-tests set; -> proj, job(name, **kw) -> job file, restore()"""
    proj = make_project(tmp)
    tr = os.path.join(proj, "code", "trainer.py")
    src = open(tr).read()
    new_src = src.replace('{"step": step, "total": 1.0}', '{"step": step, "total": 1.0, "t": time.time(), "grad_norm": 2.0 + 0.01 * step}')
    assert new_src != src
    open(tr, "w").write(new_src)
    env = {"AR_PROJECT": proj, "TS_QUEUE_EVERY": "0.5", "TS_QUEUE_IDLE": "3", "AR_HEARTBEAT_S": "0.5", "AR_STALE_S": "4", "AR_RAM_CHECKS": "1", "AR_SETTLE_S": "0",
           "AR_MACHINE_LOCK": os.path.join(tmp, "machine_lock.json")}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    jobs = os.path.join(tmp, "jobs")
    os.makedirs(jobs)

    def job(name, **kw):
        d = dict(id=name, config="base.json", set={"steps": 20, "sleep": 0.02}, no_card=True)
        d.update(kw)
        p = os.path.join(jobs, name.replace(".", "_") + ".json")
        json.dump(d, open(p, "w"))
        return p

    def restore():
        from . import trainqueue as TQ
        rp = TQ.runner_alive(TQ.context(proj))
        if rp:
            try:
                os.kill(rp, 9)
            except OSError:
                pass
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return proj, job, restore


def api_checks(tmp, ok):
    proj, job, restore = toy(tmp)
    T = api.Training()                                                 # the project of AR_PROJECT
    rec = lambda rid, name: os.path.join(proj, "records", "runs", rid, name)          # noqa: E731
    try:
        # 1 a job is validated: every key present afterwards, a typo shows, the types are checked
        j = api.load_job(job("j.one", after="a, b", priority="3", kill="group.lines@1000 > 1.48"))
        assert j["after"] == ["a", "b"] and j["priority"] == 3 and j["kill"] == ["group.lines@1000 > 1.48"] and j["set"] == ["steps=20", "sleep=0.02"] and j["kind"] == "train", j
        raises(api.load_job, "unknown job key", dict(id="x", config="b.json", stepz=1))
        raises(api.load_job, '"id"', dict(config="b.json"))
        raises(api.load_job, '"config"', dict(id="x"))
        raises(api.load_job, "priority", dict(id="x", config="b.json", priority="soon"))
        raises(api.load_job, '"card"', dict(id="x", config="b.json", card={"hypothesis": "h"}))
        raises(api.load_job, "kind", dict(id="x", config="b.json", kind="dance"))
        raises(api.load_job, "job file", os.path.join(tmp, "no_such.json"))
        raises(api.load_job, "starts with a letter", dict(id="../x", config="b.json"))
        raises(api.load_job, "unknown job key", dict(id="x", config="b.json", n=3))               # the keys of a runner job are not a training's
        raises(api.load_job, "kill", dict(id="x", config="b.json", kill=["total at 5 bigger 99"]))
        for k in ("no_card", "prepare", "no_check"):
            raises(api.load_job, "true or false", dict(id="x", config="b.json", **{k: "false"}))
        assert api.load_job(dict(id="x", config="b.json", _comment="free text"))["id"] == "x", "keys starting with _ are comments"
        p_bom = os.path.join(tmp, "bom.json")
        open(p_bom, "w", encoding="utf-8-sig").write(json.dumps(dict(id="b.om", config="base.json", no_card=True)))
        assert api.load_job(p_bom)["id"] == "b.om", "a job file with a BOM (PowerShell 5.1) is read"
        ok("jobs: normalised, a typo / a missing id / a bad type / a bad kind / true-false as text is an error, _comments are allowed, a BOM is read")

        # 2 check starts nothing and says what run would do
        r = T.check(job("c.one"))
        assert r["ok"] and r["id"] == "c.one" and any("steps" in x for x in r["changes"]) and "run.py" in r["command"] and "--id c.one" in r["command"], r
        assert not os.path.exists(os.path.join(proj, "records", "runs", "c.one")) and not os.path.exists(os.path.join(proj, "runs", "c_one")), "check starts nothing"
        raises(T.check, "needs its card", dict(id="c.two", config="base.json"))
        assert not hasattr(T, "queue_add") and not any(m.startswith("queue_") for m in api.Training.METHODS), "the queue is not part of the class"
        ok("check: ok / changes / the exact command; nothing started, no run record; a run without a card is refused; the class has no queue")

        # 3 run: start now, status reads it, the kill rules land in the run folder, a second run, compare
        r = T.run(job("r.one", kill=["total@5 > 99"]))
        assert r["id"] == "r.one" and r["started"][-1]["id"] == "r.one" and "--id r.one" in r["command"], r
        assert open(os.path.join(proj, "runs", "r_one", "kill.txt")).read().strip() == "total@5 > 99", "the job's kill rule is where watch / status read them"
        until(lambda: os.path.exists(rec("r.one", "done")), 60, what="r.one done")
        s = T.status("r.one")
        assert s["id"] == "r.one" and s["state"] == "done" and s["step"] == 20 and s["total"] == 20 and s["speed"]["s_step"] and isinstance(s["findings"], list), s
        assert [k["rule"] for k in s["kill"]] == ["total@5 > 99"] and s["kill"][0]["status"] == "ok", s["kill"]
        assert T.status("r_one")["id"] == "r.one" and T.status("r.one")["grad"]["rows"] == 20 and T.status("r.one")["grad"]["peak_step"] == 20, "the gradient line"
        assert T.status()["state"] == "idle", "nothing runs"
        raises(T.status, "no run record", "nope-xyz")                    # a typo in an id is an error
        T.run(job("r.two", set={"steps": 20, "sleep": 0.04}))
        until(lambda: os.path.exists(rec("r.two", "done")), 60, what="r.two done")
        cmpd = T.compare("r.one", "r.two")
        assert cmpd["common_step"] == 20 and [x["id"] for x in cmpd["rows"]] == ["r.one", "r.two"] and cmpd["rows"][0]["speedup"] == 1.0 and cmpd["rows"][1]["s_step"] > cmpd["rows"][0]["s_step"], cmpd
        try:
            import matplotlib  # noqa: F401
            assert T.plot("r.one", out=os.path.join(tmp, "c.png"))["png"] and os.path.getsize(os.path.join(tmp, "c.png")) > 1000
        except ImportError:
            pass
        ok("run: started with its kill rules, status (also by folder name) reads state / step / gradient / speed / kill rules, compare puts two runs side by side, a typo in an id is an error")

        # 4 stop puts the id on hold; list shows it; run refuses it; resume needs the person's words and continues from the checkpoint
        T.run(job("s.one", set={"steps": 600, "sleep": 0.05}))
        until(lambda: T.status("s.one")["step"] >= 3, 60, what="s.one running")
        assert T.status("s.one")["state"] == "running" and T.status()["id"] == "s.one", "no i = the running one"
        lst = T.list()
        assert lst["running"] == ["s.one"] and {"slots", "lock", "ram", "gpu", "holds"} <= set(lst) and "s.one" in [x["id"] for x in lst["runs"]], lst
        r = T.stop(None, "selftest")
        assert r["id"] == "s.one" and r["requested"] is True and r["verified"] is not None and r["stopped"] is True, r
        assert os.path.exists(os.path.join(proj, "runs", "s_one", "STOPPED")), "stop waited until the trainer had stopped"
        until(lambda: os.path.exists(rec("s.one", "HOLD")), 60, what="s.one stopped and held")
        assert T.status("s.one")["state"] == "stopped"
        lst = T.list()
        assert any(h["id"] == "s.one" for h in lst["holds"]) and next(x for x in lst["runs"] if x["id"] == "s.one")["held"] is True and lst["running"] == [], lst
        raises(T.stop, "nothing is running", None)
        # the request is written again while the trainer has not stopped: a fake run that never reads CONTROL, the sends are counted
        from . import control
        fake = os.path.join(proj, "runs", "fake_run")
        os.makedirs(fake)
        os.makedirs(os.path.join(proj, "records", "runs", "fake.run"))
        json.dump({"status": "running"}, open(os.path.join(proj, "records", "runs", "fake.run", "run.json"), "w"))
        sent, real_send, real_resend = [], control.request_stop, api.RESEND_S
        control.request_stop = lambda rd, why="": sent.append(why) or real_send(rd, why)
        api.RESEND_S = 0.5
        try:
            assert api._wait_stopped(TQ.context(proj), "fake.run", fake, "again", 2.5) is False
        finally:
            control.request_stop, api.RESEND_S = real_send, real_resend
        assert len(sent) >= 2, sent                                                 # written again at about 1 s and 2 s
        raises(T.run, "ON HOLD", job("s.one", set={"steps": 600, "sleep": 0.05}))
        raises(T.resume, "words", "s.one", "")
        raises(T.resume, "words", "s.one", "  ")
        step0 = T.status("s.one")["step"]
        res = T.resume("s.one", "Mirko: selftest, go on")
        assert res["id"] == "s.one" and res["cleared"] is True and res["started"] and "--pin" in res["command"], res
        assert "hold cleared" in open(os.path.join(proj, "LEDGER.csv"), encoding="utf-8").read(), "the clearing is noted in the ledger"
        until(lambda: T.status("s.one")["state"] == "running" and T.status("s.one")["step"] >= step0 + 20, 90, what="s.one goes on from its checkpoint")
        raises(T.resume, "stop it first", "s.one", "Mirko: again")
        T.stop("s.one", "selftest cleanup")
        until(lambda: os.path.exists(rec("s.one", "HOLD")), 60, what="s.one stopped again")
        ok("stop: the id goes on hold (list shows it), run refuses it, resume needs the person's words, clears the hold, continues from the checkpoint on the first code and notes it in the ledger")

        # 5 the command line says the same words
        rc, out, err = cli("check", job("k.one"), "--json")
        assert rc == 0 and json.loads(out)["ok"] is True and json.loads(out)["id"] == "k.one", (rc, out, err)
        rc, out, err = cli("check", job("k.bad", config="no_such_config.json"))
        assert rc == 2 and "error:" in err, (rc, out, err)
        rc, out, err = cli("Training.run", job("k.run", set={"steps": 20, "sleep": 0.02}), "--json")
        assert rc == 0 and json.loads(out)["id"] == "k.run", (rc, out, err)
        until(lambda: os.path.exists(rec("k.run", "done")), 60, what="k.run done")
        rc, out, err = cli("Training.status", "k_run", "--json")
        assert rc == 0 and json.loads(out)["state"] == "done", out
        rc, out, err = cli("list")
        assert rc == 0 and "k.run" in out and "slots:" in out and "machine lock:" in out, out
        rc, out, err = cli("list", "--json")
        assert rc == 0 and "k.run" in [x["id"] for x in json.loads(out)["runs"]], out
        rc, out, err = cli("Training.status", "nope-xyz")
        assert rc == 2 and "no run record" in err and "Traceback" not in err, (rc, err)
        rc, out, err = cli("Training.stop")
        assert rc == 2 and "nothing is running" in err, (rc, err)
        try:
            cli("Training.resume", "s_one")                                                 # --words is required
            raise AssertionError("resume without --words must be refused by the command line too")
        except SystemExit as e:
            assert e.code == 2
        rc, out, err = cli("Training.resume", "s_one", "--words", "")
        assert rc == 2 and "words" in err, (rc, err)
        rc, out, err = cli("methods")
        assert rc == 0 and "Training.status" in out and "run  (Training.run(job, wait=3.0))" in out and "queue" not in out, out
        rc, out, err = cli("Training.compare", "r.one", "r.two", "--skip", "2", "--json")
        assert rc == 0 and json.loads(out)["common_step"] == 20, (rc, out, err)
        for gone in ("queue-add", "queue_list", "Training.queue_hold"):
            try:
                rc, out, err = cli(gone)
                assert rc not in (0, None), (gone, rc)
            except SystemExit as e:
                assert e.code == 2, gone
        ok("command line: check / run / Training.status / list / Training.stop / Training.resume / compare say what the functions return (--json is the same dict), errors go to stderr with exit 2, the queue words are gone")

        # 6 the generator, the errors, the older words
        assert suitecli.clean({"loss": float("nan"), "xs": [1.0, float("inf")]}) == {"loss": None, "xs": [1.0, None]}
        raises(api.Training("/nonexistent/project").list, "no project")
        rc, out, err = cli("list", "--project", "/nonexistent/project")
        assert rc == 2 and "no project" in err, (rc, err)
        assert set(LEGACY) & set(api.Training.METHODS) == {"status", "stop", "plot", "compare", "resume"}, "the older words that a method also has: they need the class name"
        for word in LEGACY:                                                          # every older word is a real command (a word that is not would hide a method behind it)
            try:
                cli(word, "--help")
                raise AssertionError(word)
            except SystemExit as e:
                assert e.code == 0, word
        assert all(not m["summary"].endswith(("e.g.", "i.e.")) for m in api.Training.methods()), "a summary is a whole sentence"
        second = make_project(os.path.join(tmp, "second"))
        os.makedirs(os.path.join(second, "records", "runs", "only.second"))
        seen = {"proj": [], "second": []}

        def look(label, path):
            for _ in range(15):
                try:
                    api.Training(path).status("only.second")
                    seen[label].append(True)
                except api.TrainingError:
                    seen[label].append(False)
        threads = [threading.Thread(target=look, args=("proj", proj)), threading.Thread(target=look, args=("second", second))]
        [th.start() for th in threads]
        [th.join() for th in threads]
        assert all(seen["second"]) and not any(seen["proj"]), seen
        ok("generator: nan is null, no project is an error with exit 2, the older words are pinned and real commands, two threads on two projects do not mix")
    finally:
        restore()


def seam_checks(tmp, ok):
    """the nine defects of the seam review of 2026-11-10: each check fails on the code before the fix"""
    import signal
    import sys
    from . import compose as C
    sub = os.path.join(tmp, "seam")
    os.makedirs(sub)
    proj, job, restore = toy(sub)
    T = api.Training()
    rec = lambda rid, name: os.path.join(proj, "records", "runs", rid, name)          # noqa: E731
    pj = os.path.join(proj, "project.json")

    def project(**kw):                                                  # change the project.json of the toy (the context is read on every call)
        d = json.load(open(pj))
        for k, v in kw.items():
            if k == "prepare":
                d["trainsuite"]["prepare"] = v
            else:
                d[k] = v
        json.dump(d, open(pj, "w"))
    try:
        # 1 a trainer that was killed (no traceback, no stop file): the launcher's record says failed, so does status; the fresh heartbeat of the run folder does not say running
        T.run(job("k.kill", set={"steps": 400, "sleep": 0.05}))
        until(lambda: T.status("k.kill")["step"] >= 15, 60, what="k.kill trains")
        os.kill(json.load(open(rec("k.kill", "run.json")))["pid"], signal.SIGKILL)
        until(lambda: json.load(open(rec("k.kill", "run.json"))).get("status") == "failed", 60, what="the launcher notes the end of k.kill")
        s = T.status("k.kill")
        assert s["state"] == "failed" and s["phase"] == "failed" and any(f["code"] == "launcher_failed" and f["level"] == "alert" for f in s["findings"]), s
        assert [x["state"] for x in T.list()["runs"] if x["id"] == "k.kill"] == ["failed"], "list says the same"
        ok("status: a trainer killed without a traceback reads failed (the launcher's record), with an alert finding, as list does")

        # 2 a prepare check ends 0 present / 1 missing; any other exit is the check failing, and nothing is built on it
        probe = lambda code: [dict(name="gt", check=f"python3 -c 'import sys; sys.exit({code})'", build="python3 -c 'pass'", wait_min=1)]          # noqa: E731
        project(prepare=probe(0))
        r = T.check(job("p.zero"))
        assert r["ok"] and not r.get("will_prepare"), r
        project(prepare=probe(1))
        r = T.check(job("p.one"))
        assert not r["ok"] and any("missing data: gt" in x for x in r["problems"]), r
        r = T.check(job("p.one", prepare=True))
        assert r["ok"] and r["will_prepare"] == ["gt"], r
        project(prepare=probe(2))
        for prepare in (False, True):
            r = T.check(job("p.two", prepare=prepare))
            assert not r["ok"] and any("prepare check gt failed (exit 2" in x for x in r["problems"]) and not any("missing data" in x for x in r["problems"]) and not r.get("will_prepare"), r
        raises(T.run, "prepare check gt failed", job("p.two", prepare=True))
        assert not os.path.exists(rec("p.two.prep_gt", "run.json")) and not os.path.exists(rec("p.two", "run.json")), "no build was started on a failing check"
        project(prepare=[])
        ok("check: only exit 1 of a prepare check is 'missing data'; exit 2 is a problem that carries the output, and no build is started on it")

        # 3 a command with a quote, a $ or a space comes back from the run record as it was
        c = TQ.context(proj)
        cmd = ["bash", "-c", "$HOME/miniconda3/envs/gcar/bin/python training/gt_scenes.py x.json build"]
        r_ = dict(command=" ".join(cmd), argv=cmd, split="train", after="a.b,c.d", timeout_min=30, cwd=c.root)
        back = TQ.argv_from_record(c, "x.prep_gt", r_)
        assert back[back.index("--") + 1:] == cmd and back[back.index("--after") + 1] == "a.b,c.d" and back[back.index("--timeout") + 1] == "30" and "--cwd" not in back, back
        old = TQ.argv_from_record(c, "x.old", dict(command="python3 code/trainer.py composed/x.json", split="train"))
        assert old[old.index("--") + 1:] == ["python3", "code/trainer.py", "composed/x.json"], "a record from before `argv` is split as before"
        ok("argv_from_record: the stored argv list comes back word for word (a bash -c line with a $), with its --after / --timeout; an old record is split as before")

        # 4 a split that the project does not know is a problem of the check, not of the launcher three seconds later
        project(splits=["val", "test"])
        r = T.check(job("s.bad"))
        assert not r["ok"] and any("split 'train' is not one of this project's splits" in x for x in r["problems"]), r
        assert T.check(job("s.good", run_args=["--split", "val"]))["ok"], "the last --split of the run arguments is the one that counts"
        project(splits=["val", "train"])
        assert T.check(job("s.train"))["ok"]
        ok("check: the split of the start is checked against the project's splits")

        # 5 a queue item without a seq is numbered; a runner that fails the same way again and again ends instead of repeating it forever
        os.makedirs(c.slots, exist_ok=True)
        json.dump({"paused": None, "seq": 1, "items": [{"id": "n.seq", "argv": ["python3", "x"], "priority": 5}]}, open(c.qpath, "w"))
        q = TQ.load(c)
        assert q["items"][0]["seq"] == 2 and q["seq"] == 2, q
        lines = []
        TQ.show(c, out=lines.append)
        assert any("n.seq" in x for x in lines), lines
        json.dump({"paused": None, "seq": 5, "items": [{"id": "no.argv", "seq": 5}]}, open(c.qpath, "w"))          # an item the runner cannot start: the same KeyError at every look
        limit, TQ.ERROR_LIMIT = TQ.ERROR_LIMIT, 3
        out = []
        result = []
        th = threading.Thread(target=lambda: result.append(TQ.serve(c, every=0.05, idle_exit=60, out=out.append)), daemon=True)
        try:
            th.start()
            th.join(20)
        finally:
            TQ.ERROR_LIMIT = limit
        assert not th.is_alive() and result == [2], (th.is_alive(), result, out[-2:])
        assert sum("runner error" in x for x in out) == 3 and "the same error 3 times in a row" in out[-1], out
        assert not os.path.exists(c.runner), "the runner file is gone"
        json.dump({"paused": None, "seq": 0, "items": []}, open(c.qpath, "w"))
        ok("queue: an item without seq is numbered (runner, show); the runner ends after the same error ERROR_LIMIT times in a row, with exit 2 and its pid file removed")

        # 6 the exit code of a launcher that ran the command is the command's: a trainer that fails within the wait is a failed run, not a launcher to try again or a refusal
        plan = C.plan_start(c.root, c.block, "base.json", "q.fail", [], ["steps=5", "sleep=0.01", "fail=true"], no_card=True, ar_tools=TQ.TOOLS, records=c.records)
        it = dict(id="q.fail", argv=plan["argv"], composed=os.path.relpath(plan["composed"], c.root), config=plan["config"], seq=1)
        started, text = TQ.launch(c, it, wait=20)
        assert started and "the run failed" in text and not it.get("refused") and not it.get("launch_failures"), (started, text, it)
        assert json.load(open(rec("q.fail", "run.json")))["status"] == "failed" and TQ.state(c, it) == "failed", "the run is a failed one"
        for code, key in ((1, "launch_failures"), (2, "refused")):                    # the launcher itself failing / refusing stays what it was
            it2 = dict(id=f"q.l{code}", argv=["python3", "-c", f"import sys; sys.exit({code})", "--", "x"], seq=2)
            started, text = TQ.launch(c, it2, wait=20)
            assert not started and it2.get(key), (code, started, text, it2)
        ok("launch: a trainer that fails within the wait is a started and failed run; a launcher that exits 1 is tried again, exit 2 is a refusal")

        # 7 `trainsuite curves` without matplotlib is a message and exit 2, as Training.plot says
        T.run(job("c.run", set={"steps": 10, "sleep": 0.01}))
        until(lambda: os.path.exists(rec("c.run", "done")), 60, what="c.run done")
        saved = sys.modules.get("matplotlib", "absent")
        sys.modules["matplotlib"] = None                                    # an import of it raises ImportError, installed or not
        try:
            rc, o, e = cli("--root", os.path.join(proj, "runs"), "curves", "c_run")
        finally:
            if saved == "absent":
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = saved
        assert rc == 2 and "plot needs matplotlib" in e and "Traceback" not in e, (rc, o, e)
        ok("curves: without matplotlib the legacy command says so and ends with 2")

        # 8 no wiring to the commented-out runsuite: a job of kind run is refused with the way to the Runner, a runner command is no training
        for bad in (dict(kind="run"), dict(mode="full")):
            try:
                api.load_job(dict(id="r.x", config="base.json", **bad))
                raise AssertionError(f"{bad} must be refused")
            except api.TrainingError as e:
                assert ("python -m runner" in str(e)) if "kind" in bad else ("unknown job key" in str(e)), str(e)
        assert not hasattr(api, "RUN_ONLY") and api.KINDS == ("train",)
        runner_cmd = "bash -c $HOME/miniconda3/envs/gcar/bin/python -m runner predict runs/t20_base_s0 --out runs/t20_base_s0/preds"
        assert TQ.is_training(c, "t20.base.s0", {"command": runner_cmd}) is False, "a runner job whose id is the network's folder name is no training"
        assert TQ.is_training(c, "t20.base.s0", {"command": "python3 code/trainer.py composed/t20_base_s0.json"}) is True
        ok("jobs: kind run points to `python -m runner`; a runner command is recognised by what it starts")

        # 9 a resume without a first commit stops (and leaves the hold) instead of training on the working tree of now
        T.run(job("n.pin", set={"steps": 600, "sleep": 0.05}))
        until(lambda: T.status("n.pin")["step"] >= 10, 60, what="n.pin trains")
        T.stop("n.pin", "selftest")
        until(lambda: os.path.exists(rec("n.pin", "HOLD")) and json.load(open(rec("n.pin", "run.json"))).get("status") != "running", 60, what="n.pin stopped")
        r_ = json.load(open(rec("n.pin", "run.json")))
        r_.pop("tag", None)
        json.dump(r_, open(rec("n.pin", "run.json"), "w"))
        raises(T.resume, "names no commit of the first attempt", "n.pin", "Mirko: selftest")
        assert os.path.exists(rec("n.pin", "HOLD")), "the refusal changed nothing: the hold stays"
        res = T.resume("n.pin", "Mirko: selftest, on the code of now", fresh_code=True)
        assert res["cleared"] is True and "--pin" not in res["command"], res
        until(lambda: T.status("n.pin")["state"] == "running", 60, what="n.pin goes on")
        T.stop("n.pin", "selftest cleanup")
        until(lambda: os.path.exists(rec("n.pin", "HOLD")), 60, what="n.pin stopped again")
        ok("resume: without a commit of the first attempt it is refused (the hold stays); fresh_code continues on the code of now")
    finally:
        restore()


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")
    with tempfile.TemporaryDirectory() as tmp:
        api_checks(tmp, ok)
        seam_checks(tmp, ok)
    print(f"interface selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

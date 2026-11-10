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
        raises(api.load_job, "belong to a runner job", dict(id="x", config="b.json", n=3))
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


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")
    with tempfile.TemporaryDirectory() as tmp:
        api_checks(tmp, ok)
    print(f"interface selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

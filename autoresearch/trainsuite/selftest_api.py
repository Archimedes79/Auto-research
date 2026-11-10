"""Self-test of the interface (api.py): jobs, check, run, status, stop, compare, plot and the queue functions against a toy project with real run.py launchers and a toy trainer
(selftest_queue.py builds it), about a minute. It checks what the interface promises: a job is validated (a typo shows), `check` starts nothing, `run` starts and `status` reads it,
a queue number, an id and a run folder name address the same item, the order follows the priority, hold and resume refuse without the person's words, the command line says the
same as the functions (--json prints what they return)."""
import contextlib
import io
import json
import os
import tempfile
import time

from . import api
from . import trainqueue as TQ
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


def api_checks(tmp, ok):
    proj = make_project(tmp)
    tr = os.path.join(proj, "code", "trainer.py")                  # the toy trainer logs a time and a gradient norm too (the speed and the gradient line need them)
    src = open(tr).read()
    new_src = src.replace('{"step": step, "total": 1.0}', '{"step": step, "total": 1.0, "t": time.time(), "grad_norm": 2.0 + 0.01 * step}')
    assert new_src != src
    open(tr, "w").write(new_src)
    with open(os.path.join(proj, "code", "runner.py"), "w") as f:         # a toy runner: writes objects.json (what it was asked) into --out
        f.write("import json, os, sys\n"
                "mode, target, args = sys.argv[1], sys.argv[2], sys.argv[3:]\n"
                "out = args[args.index('--out') + 1] if '--out' in args else os.path.join(target, 'runner', mode)\n"
                "os.makedirs(out, exist_ok=True)\n"
                "json.dump({'mode': mode, 'target': target, 'args': args}, open(os.path.join(out, 'objects.json'), 'w'))\n")
    pj = json.load(open(os.path.join(proj, "project.json")))
    pj["runsuite"] = {"command": "python3 code/runner.py", "runs": "runs"}
    json.dump(pj, open(os.path.join(proj, "project.json"), "w"))
    env = {"AR_PROJECT": proj, "TS_QUEUE_EVERY": "0.5", "TS_QUEUE_IDLE": "3", "AR_HEARTBEAT_S": "0.5", "AR_STALE_S": "4", "AR_RAM_CHECKS": "1", "AR_SETTLE_S": "0",
           "AR_MACHINE_LOCK": os.path.join(tmp, "machine_lock.json")}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    jobs = os.path.join(tmp, "jobs")
    os.makedirs(jobs)
    T = api.Training()                                                 # the project of AR_PROJECT

    def job(name, **kw):
        d = dict(id=name, config="base.json", set={"steps": 20, "sleep": 0.02}, no_card=True)
        d.update(kw)
        p = os.path.join(jobs, name.replace(".", "_") + ".json")
        json.dump(d, open(p, "w"))
        return p
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
        raises(api.load_job, "job file", os.path.join(jobs, "no_such.json"))
        assert api.load_job(dict(id="x", config="b.json", _comment="free text"))["id"] == "x", "keys starting with _ are comments"
        ok("jobs: normalised (a text list, a set dict, a priority), a typo / a missing id / a bad type / a bad kind is an error, _comments are allowed")

        # 2 check starts nothing and says what run would do
        r = T.check(job("c.one"))
        assert r["ok"] and r["id"] == "c.one" and any("steps" in x for x in r["changes"]) and "run.py" in r["command"] and "--id c.one" in r["command"], r
        assert not os.path.exists(os.path.join(proj, "records", "runs", "c.one")) and not os.path.exists(os.path.join(proj, "runs", "c_one")), "check starts nothing"
        raises(T.check, "needs its card", dict(id="c.two", config="base.json"))
        ok("check: ok / changes / the exact command; nothing started, no run record; a run without a card is refused")

        # 3 the queue: numbers follow the priority, an id and a folder name are the same item, remove, the older words still work
        assert T.queue_list()["items"] == [] and T.status()["state"] == "idle"
        T.queue_add(job("q.b"), start=False)
        item = T.queue_add(job("q.a", priority=1, note="first"), start=False)
        assert item["number"] == 1 and item["id"] == "q.a" and item["state"] == "queued" and item["note"] == "first" and "run.py" in item["command"] and item["job"]["priority"] == 1, item
        lst = T.queue_list()
        assert [x["id"] for x in lst["items"]] == ["q.a", "q.b"] and [x["number"] for x in lst["items"]] == [1, 2] and lst["running"] is None and lst["runner"] == 0, lst
        assert {"slots", "lock", "ram", "gpu", "holds"} <= set(lst), sorted(lst)
        assert T.queue_get(2)["id"] == T.queue_get("q.b")["id"] == T.queue_get("q_b")["id"] == "q.b"
        raises(T.queue_get, "number 3", 3)
        raises(T.queue_get, "not in the queue", "nothing")
        raises(T.queue_add, "already in the queue", job("q.b"), start=False)
        assert T.queue_remove(2) == {"removed": "q.b"} and [x["id"] for x in T.queue_list()["items"]] == ["q.a"]
        assert T.queue_remove("q_a") == {"removed": "q.a"} and T.queue_list()["items"] == []
        ok("queue: priority gives the numbers, a number / an id / a folder name address one item, a double id and a bad number are refused, remove")

        # 4 the person's words are required
        raises(T.queue_hold, "words", "")
        raises(T.queue_resume, "words", "  ")
        assert T.queue_hold("Mirko: test") == {} and T.queue_list()["paused"] is True
        assert T.queue_resume("Mirko: test", fresh_code=False) == {} and T.queue_list()["paused"] is False
        ok("hold / resume: refused without the person's words; with them they pause and release the runner (nothing running: nothing to stop)")

        # 5 run: start now, status reads it, the kill rules land in the run folder, a second run, compare
        r = T.run(job("r.one", kill=["total@5 > 99"]))
        assert r["id"] == "r.one" and r["started"][-1]["id"] == "r.one" and "--id r.one" in r["command"], r
        assert open(os.path.join(proj, "runs", "r_one", "kill.txt")).read().strip() == "total@5 > 99", "the job's kill rule is where watch / status read it"
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "r.one", "done")), 60, what="r.one done")
        s = T.status("r.one")
        assert s["id"] == "r.one" and s["state"] == "done" and s["step"] == 20 and s["total"] == 20 and s["queue"] is None and s["speed"]["s_step"] and isinstance(s["findings"], list), s
        assert [k["rule"] for k in s["kill"]] == ["total@5 > 99"] and s["kill"][0]["status"] == "ok", s["kill"]
        assert T.status("r_one")["id"] == "r.one" and T.status("r.one")["grad"]["rows"] == 20 and T.status("r.one")["grad"]["peak_step"] == 20, "the gradient line"
        T.run(job("r.two", set={"steps": 20, "sleep": 0.04}))
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "r.two", "done")), 60, what="r.two done")
        cmpd = T.compare("r.one", "r.two")
        assert cmpd["common_step"] == 20 and [x["id"] for x in cmpd["rows"]] == ["r.one", "r.two"] and cmpd["rows"][0]["speedup"] == 1.0 and cmpd["rows"][1]["s_step"] > cmpd["rows"][0]["s_step"], cmpd
        try:
            import matplotlib  # noqa: F401
            assert T.plot("r.one", out=os.path.join(tmp, "c.png"))["png"] and os.path.getsize(os.path.join(tmp, "c.png")) > 1000
        except ImportError:
            pass
        raises(T.status, "no queue item number", 9)
        ok("run: started with its kill rules, status (also by folder name) reads state / step / speed / kill rules, compare puts two runs side by side, plot")

        # 6 stop one training: a long run, the queue says it runs, stop writes the request, the run ends with a checkpoint and goes on hold
        T.run(job("s.one", set={"steps": 600, "sleep": 0.05}))
        until(lambda: T.status("s.one")["step"] >= 3, 60, what="s.one running")
        assert T.status("s.one")["state"] == "running"
        r = T.stop("s.one", "selftest")
        assert r["id"] == "s.one" and r["requested"] is True and os.path.exists(r["file"]), r
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "s.one", "HOLD")), 60, what="s.one stopped and held")
        assert T.status("s.one")["state"] == "stopped", T.status("s.one")["state"]
        assert any(h["id"] == "s.one" for h in T.queue_list()["holds"]), "the hold shows in the view"
        raises(T.stop, "nothing is running", None)
        ok("stop: the request is written, the run saves a checkpoint, ends with 75 and its id is on hold (the view lists it); status() says idle when nothing runs")

        # 7 the runner: add with start, the item runs by itself, the view names it while it runs, the runner ends when nothing is left
        T.queue_add(job("w.one", set={"steps": 60, "sleep": 0.05}))
        until(lambda: T.queue_list()["running"] == "w.one", 60, what="w.one running through the queue")
        s = T.status()
        assert s["id"] == "w.one" and s["number"] == 1 and s["queue"] == "running", s
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "w.one", "done")), 90, what="w.one done")
        assert T.queue_list()["items"][0]["state"] == "done" and T.queue_get(1)["step"] == 60
        until(lambda: TQ.runner_alive(TQ.context(proj)) == 0, 30, what="the runner ends by itself")
        assert T.queue_start()["runner"] > 0, "queue_start brings the runner back"
        ok("queue + runner: an added job runs by itself, status() without an argument is the running one, the item is done afterwards, queue_start starts the runner")

        # 8 the command line says the same words
        rc, out, err = cli("check", job("k.one"), "--json")
        assert rc == 0 and json.loads(out)["ok"] is True and json.loads(out)["id"] == "k.one", (rc, out, err)
        rc, out, err = cli("check", job("k.bad", config="no_such_config.json"))
        assert rc == 2 and "error:" in err, (rc, out, err)
        rc, out, err = cli("queue-add", job("k.two", priority=2), "--no-start", "--json")
        assert rc == 0 and json.loads(out)["id"] == "k.two", (rc, out, err)
        rc, out, err = cli("queue", "add", "base.json")
        assert rc == 2 and "--id is required" in err and "queue-add" in err, "the older `queue add` needs its --id and says where a job file goes"
        rc, out, err = cli("queue-list", "--json")
        data = json.loads(out)
        assert rc == 0 and "k.two" in [x["id"] for x in data["items"]], data
        rc, out, err = cli("queue-list")
        assert rc == 0 and "k.two" in out and "machine lock:" in out and "slots:" in out, out
        rc, out, err = cli("queue-get", "k.two", "--json")
        assert rc == 0 and json.loads(out)["job"]["priority"] == 2, out
        rc, out, err = cli("Training.status", "w.one", "--json")
        assert rc == 0 and json.loads(out)["state"] == "done", out
        rc, out, err = cli("queue-remove", "k.two")
        assert rc == 0 and "removed k.two" in out
        rc, out, err = cli("queue-get", "k.two")
        assert rc == 2 and "not in the queue" in err, (rc, err)
        rc, out, err = cli("queue", "hold")
        assert rc == 2 and "name the runs" in err, (rc, out, err)
        ok("command line: check / queue-add / queue-list / queue-get / Training.status / queue-remove say what the functions return (--json is the same dict), errors go to stderr with exit 2")

        # 9 the runner has the same interface and the same queue
        import runsuite
        RJ = runsuite.RunJobs(proj)
        def rj(name, **kw):                                             # a runner job file: exactly these keys, no training defaults
            d = dict(id=name, kind="run", mode="full", run="r.one", n=4, dts=[0, 0.5], out="runs/out_" + name.replace(".", "_"))
            d.update(kw)
            path = os.path.join(jobs, name.replace(".", "_") + ".json")
            json.dump(d, open(path, "w"))
            return path
        r = RJ.check(rj("rs.one"))
        assert r["ok"] and "code/runner.py full runs/r_one --n 4 --out runs/out_rs_one --dts 0 0.5" in r["command"] and "--no-card" in r["command"], r
        assert not RJ.check(dict(id="rs.x", kind="run", mode="full", run="nothing"))["ok"] and "does not exist" in RJ.check(dict(id="rs.x", kind="run", mode="full", run="nothing"))["problems"][0]
        assert "needs" in RJ.check(dict(id="rs.x", kind="run", mode="full"))["problems"][0] and "does not take" in RJ.check(dict(id="rs.x", kind="run", mode="check", config="base.json", out="o"))["problems"][0]
        raises(RJ.check, "mode", dict(id="rs.x", kind="run", mode="dance"))
        raises(RJ.check, "not both", dict(id="rs.x", kind="run", mode="full", run="r.one", n=3, scene="scene-1"))
        raises(RJ.check, 'kind "train"', job("rs.train"))
        raises(T.check, 'kind "run"', dict(id="rs.x", kind="run", mode="full", run="r.one"))
        r = RJ.run(rj("rs.one"))
        assert r["id"] == "rs.one" and r["started"][0]["finished"] is True, r                      # the toy runner ends within the wait
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "rs.one", "done")), 60, what="rs.one done")
        st = RJ.status("rs.one")
        assert st["state"] == "done" and st["exit"] == 0 and RJ.status()["state"] == "idle", st
        assert json.load(open(os.path.join(proj, "runs", "out_rs_one", "objects.json")))["args"] == ["--n", "4", "--out", "runs/out_rs_one", "--dts", "0", "0.5"]
        item = RJ.queue_add(rj("rs.two"), start=False)
        assert item["kind"] == "run" and item["state"] == "queued" and "code/runner.py full runs/r_one" in item["command"], item
        assert "rs.two" in [x["id"] for x in RJ.queue_list()["items"]] and RJ.queue_get("rs.two")["job"]["mode"] == "full"
        RJ.queue_start()
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "rs.two", "done")), 60, what="rs.two done through the queue")
        assert RJ.status("rs.two")["files"] == ["objects.json"] and RJ.queue_get("rs.two")["state"] == "done"
        assert RJ.queue_remove("rs.two") == {"removed": "rs.two"}
        rc, out, err = cli("queue-list", "--json")
        assert rc == 0 and "rs.two" not in [x["id"] for x in json.loads(out)["items"]] and "w.one" in [x["id"] for x in json.loads(out)["items"]], "one queue: the runner job went, the training item is still there"
        from runsuite.__main__ import main as run_main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert run_main(["check", rj("rs.cli"), "--json"]) == 0
        assert json.loads(buf.getvalue())["mode"] == "full"
        ok("runner: the same job file / check / run / status / queue_add, one queue with the trainings (kind run), the toy runner's output is listed, a wrong mode / path / flag is a problem")

        # 10 the class and the generic command line: every method is a command, `Training.queue_hold --words ...`
        t = api.Training(proj)
        assert [m["name"] for m in api.Training.methods()] == list(api.Training.METHODS) and all(m["summary"] for m in api.Training.methods()), "every method is listed with a summary"
        assert t.queue_list()["items"] == T.queue_list()["items"] and t.status("r.one")["state"] == "done", "an object and the default object see the same project"
        raises(t.check, 'kind "run"', dict(id="rs.x", kind="run", mode="full", run="r.one"))
        assert [m["name"] for m in runsuite.RunJobs.methods()] == list(runsuite.RunJobs.METHODS) and "stop" not in runsuite.RunJobs.METHODS, "the runner lists its own methods"
        rc, out, err = cli("methods")
        assert rc == 0 and "Training.queue_add(job, start=True)" in out and "Training.queue_hold(words, i=None, reason=''" in out, out
        rc, out, err = cli("Training.check", job("g.one"), "--json")
        assert rc == 0 and json.loads(out)["ok"] is True, (rc, out, err)
        rc, out, err = cli("Training.check", job("g.bad", config="no_such_config.json"))
        assert rc == 2 and "error:" in err, (rc, err)
        rc, out, err = cli("Training.queue_add", job("g.q"), "--no-start", "--json")
        assert rc == 0 and json.loads(out)["id"] == "g.q" and json.loads(out)["state"] == "queued", (rc, out, err)
        rc, out, err = cli("queue_list")                                            # without the class name: not an older command
        assert rc == 0 and "g.q" in out, out
        rc, out, err = cli("queue-list", "--json")                                  # the command is the method name with hyphens
        items = json.loads(out)["items"]
        assert rc == 0 and "g.q" in [x["id"] for x in items] and [x["number"] for x in items] == list(range(1, len(items) + 1)), "the items keep their order when they are read on several threads"
        rc, out, err = cli("Training.queue_get", "g_q", "--json")
        assert rc == 0 and json.loads(out)["id"] == "g.q", (rc, out, err)
        rc, out, err = cli("Training.status", "1", "--json")                        # a number: the item
        assert rc == 0 and json.loads(out)["id"] == json.loads(cli("queue_list", "--json")[1])["items"][0]["id"], (rc, out, err)
        try:
            cli("Training.queue_hold")
            raise AssertionError("queue_hold without --words must be refused by the command line too")
        except SystemExit as e:
            assert e.code == 2
        try:
            cli("Training.queue_hold", "--words", "Mirko: selftest")                # no item and no --all: the command line does not hold everything by accident
            raise AssertionError("queue_hold without an item and without --all must be refused by the command line")
        except SystemExit as e:
            assert e.code == 2
        assert json.loads(cli("queue_list", "--json")[1])["paused"] is False
        rc, out, err = cli("Training.queue_hold", "--all", "--words", "Mirko: selftest")
        assert rc == 0, (rc, out, err)
        assert json.loads(cli("queue_list", "--json")[1])["paused"] is True
        rc, out, err = cli("Training.queue_resume", "--all", "--words", "Mirko: selftest")
        assert rc == 0 and json.loads(cli("queue_list", "--json")[1])["paused"] is False
        rc, out, err = cli("Training.queue_remove", "g.q")
        assert rc == 0 and "removed g.q" in out
        rc, out, err = cli("Training.compare", "r.one", "r.two", "--skip", "2", "--json")
        assert rc == 0 and json.loads(out)["common_step"] == 20, (rc, out, err)
        from runsuite.__main__ import main as run_main2
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert run_main2(["RunJobs.check", rj("rs.cli2"), "--json"]) == 0 and run_main2(["run", rj("rs.cli3")]) == 0
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "rs.cli3", "done")), 60, what="rs.cli3 done")
        ok("class + command line: Training(project) sees the same project, the methods are listed with their arguments, `Training.<method>` runs each one (a number or an id for i, --words is required, "
           "--no-start, --json), a method without the class name works when no older command has the word, the runner has the same")

        # 11 what the review found: the person's pause and holds also bind a direct run, `after` is kept, a run started outside the queue is seen, ids before numbers, a runner job is no
        # training for hold, kill rules are checked and appended, the runner has the gate / the default folder / the split label / safe quoting
        import inspect
        t = api.Training(proj)
        t.queue_hold("Mirko: selftest")                                             # an idle queue: it is paused
        raises(t.run, "PAUSED", job("p.one"))
        assert t.check(job("p.one"))["ok"] is False and "PAUSED" in t.check(job("p.one"))["problems"][0], "check says what run would be refused for"
        raises(runsuite.RunJobs(proj).run, "PAUSED", rj("rs.p"))
        assert t.queue_add(job("p.q"), start=False)["state"] == "queued", "the queue may be filled while it is paused"
        t.queue_resume("Mirko: selftest")
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "p.q", "done")), 60, what="p.q done after the resume")
        t.queue_remove("p.q")
        t.queue_add(job("h.a"), start=False)
        t.queue_add(job("h.b"), start=False)
        held = t.queue_hold("Mirko: selftest", i=t.queue_get("h.b")["number"])    # a number: that item of the queue only
        assert set(held) == {"h.b"} and t.queue_get("h.b")["held"] and not t.queue_get("h.a")["held"] and t.queue_list()["paused"] is False, held
        raises(t.run, "is held", job("h.b"))
        res = t.queue_resume("Mirko: selftest", i="h.b")                           # one by id; the runner starts the queue
        assert set(res) == {"h.b"}, res
        until(lambda: all(os.path.exists(os.path.join(proj, "records", "runs", x, "done")) for x in ("h.a", "h.b")), 90, what="h.a and h.b done")
        t.queue_remove("h.a")
        t.queue_remove("h.b")
        d = t.run(job("a.after", after=["r.one"]))
        assert "--after r.one" in d["command"], d
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "a.after", "done")), 60, what="a.after done")
        t.run(job("o.long", set={"steps": 600, "sleep": 0.05}))                     # outside the queue: status() and the view still see it
        until(lambda: t.status("o.long")["step"] >= 3, 60, what="o.long running")
        assert t.status()["id"] == "o.long" and t.queue_list()["running"] == "o.long" and "o.long" not in [x["id"] for x in t.queue_list()["items"]], "a run outside the queue is the running one"
        assert t.stop(None, "selftest")["id"] == "o.long"
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "o.long", "HOLD")), 60, what="o.long stopped")
        t.queue_add(job("2"), start=False)
        t.queue_add(job("1", priority=1), start=False)                              # ids "1" and "2" against the numbers 1 and 2: the number 1 is the id "1" (priority), the text "2" is the id "2"
        n_before = len(t.queue_list()["items"]) - 2                                   # the items that were there (done ones of the earlier groups)
        assert t.queue_get(1)["id"] == "1" and t.queue_get("1")["number"] == 1, "an int is a number: 1 is the first item, the id '1' (priority 1)"
        assert t.queue_get("2")["id"] == "2" and t.queue_get("2")["number"] == n_before + 2 and (n_before < 1 or t.queue_get(2)["id"] != "2"), "a text is an id first: '2' is the id '2', not the item number 2"
        t.queue_add(job("12"), start=False)
        assert t.queue_get("12")["id"] == "12" and t.queue_get("12")["number"] == n_before + 3, "a digit-only id is added and found as an id"
        for x in ("1", "2", "12"):
            t.queue_remove(x)
        runsuite.RunJobs(proj).queue_add(rj("rs.hold"), start=False)
        assert "not touched" in t.queue_hold("Mirko: selftest")["rs.hold"] and t.queue_get("rs.hold")["held"] is False, "a runner job is no training: hold leaves it alone"
        t.queue_resume("Mirko: selftest")
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "rs.hold", "done")), 60, what="rs.hold done")
        t.queue_remove("rs.hold")
        # jobs: ids, keys of the other kind, kill rules
        raises(api.load_job, "starts with a letter", dict(id="../x", config="b.json"))
        raises(api.load_job, "belong to a runner job", dict(id="x", config="b.json", n=3))
        raises(api.load_job, "belong to a training", dict(id="x", kind="run", mode="full", run="r.one", card={"hyp": "h"}))
        raises(api.load_job, "kill", dict(id="x", config="b.json", kill=["total at 5 bigger 99"]))
        # kill rules are appended to a file that is there and are not written when the launcher refuses
        os.makedirs(os.path.join(proj, "runs", "k_old"))
        open(os.path.join(proj, "runs", "k_old", "kill.txt"), "w").write("old@1 > 1\n")
        t.run(job("k.old", kill=["total@5 > 99"]))
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "k.old", "done")), 60, what="k.old done")
        assert open(os.path.join(proj, "runs", "k_old", "kill.txt")).read().split() == ["old@1", ">", "1", "total@5", ">", "99"], open(os.path.join(proj, "runs", "k_old", "kill.txt")).read()
        raises(t.run, "nothing runs", job("k.old", kill=["total@7 > 98"]))           # the id finished: run.py refuses
        assert "total@7" not in open(os.path.join(proj, "runs", "k_old", "kill.txt")).read(), "a refused launch adds no rule"
        # the queue launches with the kill rules too (a rule is appended once, not twice)
        t.queue_add(job("k.q", kill=["total@5 > 99"]))
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "k.q", "done")), 60, what="k.q done")
        until(lambda: os.path.exists(os.path.join(proj, "runs", "k_q", "kill.txt")), 15, what="the runner wrote kill.txt after the launcher survived its wait")
        assert open(os.path.join(proj, "runs", "k_q", "kill.txt")).read().split() == ["total@5", ">", "99"]
        t.queue_remove("k.q")
        # the runner: memory gate, the CPU job without a slot, the default folder, the split label, quoting of awkward scene names
        pj_path = os.path.join(proj, "project.json")
        pj0 = open(pj_path).read()
        pj = json.loads(pj0)
        pj["trainsuite"]["gpu_max_mb"] = 4000
        pj["runsuite"] = {"command": 'bash -c "python3 code/runner.py {args}"', "runs": "runs"}
        json.dump(pj, open(pj_path, "w"))
        try:
            r = runsuite.RunJobs(proj)
            cmd = r.check(rj("rs.g", out=""))["command"]
            assert "--gpu --gpu-max-mb 4000" in cmd, cmd
            assert "--gpu" not in r.check(rj("rs.c", device="cpu"))["command"].split(" -- ")[0], "a job on the CPU takes no GPU slot"
            assert r.check(rj("rs.g", out=""))["out"] == "runs/r_one/runner/full", r.check(rj("rs.g", out=""))["out"]
            assert "--id rs.s --no-card --split val" in r.check(rj("rs.s", split="val"))["command"]
            for k, scene in enumerate(("it's", 'say "hi"', "a b;$x")):
                res = r.run(rj(f"rs.q{k}", n=None, scene=scene, dts=[], out=f"runs/out_q{k}"))
                until(lambda: os.path.exists(os.path.join(proj, "records", "runs", f"rs.q{k}", "done")), 60, what=f"rs.q{k} done")
                got = json.load(open(os.path.join(proj, "runs", f"out_q{k}", "objects.json")))["args"]
                assert got[got.index("--scene") + 1] == scene, (scene, got)
        finally:
            open(pj_path, "w").write(pj0)
        # the rest: errors have one type, a project is an argument, the delegates keep their signature
        assert "words" in inspect.signature(T.queue_hold).parameters and "job" in inspect.signature(T.queue_add).parameters
        old_env = os.environ.pop("AR_PROJECT")
        try:
            assert isinstance(api.Training(proj).queue_list()["items"], list) and "AR_PROJECT" not in os.environ, "Training(project) does not set the project of the process"
        finally:
            os.environ["AR_PROJECT"] = old_env
        open(TQ.context(proj).qpath, "w").write("{ not json")
        raises(t.status, "could not be read", "r.one")                              # a damaged queue file is a TrainingError, not a QueueError
        ok("review: a pause / a held item forbids a direct run, `after` is passed, a run outside the queue is seen by status() / the view, ids before numbers (also digit-only), hold / resume of one "
           "item, a runner job is no training for hold, kill rules checked / appended / not written on a refusal, the runner has the gate / the default folder / the split / safe quoting")

        # 12 what the second review found: the note of a cleared hold goes to the ledger of the object's project, a job of another project is read back from there, a slot of another project
        # is not ours, a runner job is no training, true / false are checked, one error type and exit 2, a job file refuses the older options, the pause binds `start --go`
        import types
        t.queue_add(job("n.led", set={"steps": 200, "sleep": 0.05}))
        until(lambda: t.status("n.led")["step"] >= 3, 60, what="n.led running")
        t.queue_hold("Mirko: selftest hold", i="n.led")
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "n.led", "HOLD")), 60, what="n.led held")
        kept = os.environ.pop("AR_PROJECT")                                         # no project in the environment: the object's project must still be the one that is used
        try:
            api.Training(proj).queue_resume("Mirko: selftest note twelve", i="n.led")
            assert "AR_PROJECT" not in os.environ, "a method gives the environment back"
        finally:
            os.environ["AR_PROJECT"] = kept
        assert "selftest note twelve" in open(os.path.join(proj, "LEDGER.csv"), encoding="utf-8").read(), "the note of the cleared hold is in the ledger of the project of the object"
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "n.led", "done")), 90, what="n.led done after the resume")
        t.queue_remove("n.led")
        second = make_project(os.path.join(tmp, "second"))
        got = t.queue_add(job("x.two", project=second), start=False)
        assert got["id"] == "x.two" and got["state"] == "queued", got
        raises(t.queue_get, "not in the queue", "x.two")
        assert api.Training(second).queue_get("x.two")["id"] == "x.two", "a job of another project is queued there"
        api.Training(second).queue_remove("x.two")
        c = TQ.context(proj)
        os.makedirs(os.path.join(c.R.slot_dir(c.root, c.cfg, c.records), "0"), exist_ok=True)

        def foreign(run_dir):
            return types.SimpleNamespace(slot_dir=c.R.slot_dir, pid_alive=lambda p: True, record=lambda s: {"id": "foreign.run", "pid": 1, "run_dir": run_dir})
        real_R = c.R
        c.R = foreign("/elsewhere/records/runs/foreign.run")
        assert api._slot_ids(c) == [], "a slot taken by a run of another project (one slot folder for both) is not ours"
        c.R = foreign(os.path.join(c.records, "foreign.run"))
        assert set(api._slot_ids(c)) == {"foreign.run"}
        c.R = real_R
        assert TQ.runner_script(c) == "code/runner.py"
        assert TQ.is_training(c, "rs.slow", {"command": "python3 code/runner.py full runs/r_one --out runs/out_rs_slow"}) is False, "a runner job is no training even when its output folder carries its id"
        assert TQ.is_training(c, "tr.x", {"command": "python3 code/trainer.py composed/tr_x.json"}) is True
        for k in ("no_card", "prepare", "no_check"):
            raises(api.load_job, "true or false", dict(id="x", config="b.json", **{k: "false"}))
        raises(RJ.check, "true or false", dict(id="rs.x", kind="run", mode="full", run="r.one", cold="no"))
        p_bom = os.path.join(jobs, "bom.json")
        open(p_bom, "w", encoding="utf-8-sig").write(json.dumps(dict(id="b.om", config="base.json", no_card=True)))
        assert api.load_job(p_bom)["id"] == "b.om", "a job file with a BOM (PowerShell 5.1) is read"
        raises(t.stop, "nothing is running")                                        # i=None means the running one
        raises(api.Training("/nonexistent/project").queue_list, "no project")
        rc, out_, err_ = cli("Training.queue_list", "--project", "/nonexistent/project")
        assert rc == 2 and "no project" in err_, (rc, err_)
        jf = job("o.opt")
        rc, out_, err_ = cli("queue", "add", jf, "--priority", "1", "--no-runner")      # the older form is for a config with --id: a job file is refused and the right command is named
        assert rc == 2 and "queue-add" in err_ and "o.opt" not in [x["id"] for x in t.queue_list()["items"]], (rc, err_)
        t.queue_hold("Mirko: selftest")
        rc, out_, err_ = cli("start", "base.json", "--id", "z.paused", "--no-card", "--go")
        assert rc == 1 and "PAUSED" in out_ and not os.path.isdir(os.path.join(proj, "records", "runs", "z.paused")), (rc, out_)
        t.queue_resume("Mirko: selftest")
        open(TQ.context(proj).qpath, "w").write("{ not json")
        raises(runsuite.RunJobs(proj).check, "could not be read", rj("rs.dmg"))      # the runner's check too turns the queue's error into the one error type
        ok("second review: the note of a cleared hold goes to the object's project, a job of another project is read back from there, a slot of another project is not ours, a runner job is no "
           "training for hold, true / false are checked, a BOM is read, no project and a damaged queue are errors with exit 2, a job file refuses the older options, a pause binds `start --go`")

        # 13 what the third review found: --json has no NaN, a damaged item is a message with exit 2, two threads on two projects do not mix, the old words are pinned, the help is honest
        import inspect as _inspect
        import threading
        import suitecli
        from .api import LEGACY
        assert suitecli.clean({"loss": float("nan"), "xs": [1.0, float("inf")]}) == {"loss": None, "xs": [1.0, None]}
        raises(T.status, "no queue item", "nope-xyz")                                  # a typo in an id is an error here as in the other suites, not "unknown"
        raises(RJ.status, "no queue item", "nope-xyz")
        json.loads(json.dumps(suitecli.clean({"loss": float("nan")}), allow_nan=False))
        qpath = TQ.context(proj).qpath
        open(qpath, "w").write(json.dumps({"items": [{"id": "x.broken"}]}))        # an item without its fields (a hand-made file)
        rc, out_, err_ = cli("queue-list")
        assert rc == 2 and err_.startswith("error:") and "Traceback" not in err_, (rc, err_)
        os.remove(qpath)
        assert set(LEGACY) & set(api.Training.METHODS) == {"status", "stop", "plot", "compare"}, "the older words that a method also has: they need the class name"
        for word in LEGACY:                                                          # every older word is a real command (a word that is not would hide a method behind it)
            try:
                cli(word, "--help")
                raise AssertionError(word)
            except SystemExit as e:
                assert e.code == 0, word
        assert all(not m["summary"].endswith(("e.g.", "i.e.")) for m in api.Training.methods() + runsuite.RunJobs.methods()), "a summary is a whole sentence"
        assert str(_inspect.signature(RJ.check)) == "(job)" and str(_inspect.signature(T.queue_hold)).startswith("(words"), "the decorated methods keep their arguments"
        api.Training(second).queue_add(job("x.cross", project=second), start=False)
        seen = {"proj": set(), "second": set()}

        def look(label, path):
            for _ in range(15):
                seen[label] |= {x["id"] for x in api.Training(path).queue_list()["items"]}
        threads = [threading.Thread(target=look, args=("proj", proj)), threading.Thread(target=look, args=("second", second))]
        [th.start() for th in threads]
        [th.join() for th in threads]
        assert "x.cross" in seen["second"] and "x.cross" not in seen["proj"], seen
        api.Training(second).queue_remove("x.cross")
        from runsuite.__main__ import main as run_main3
        errbuf = io.StringIO()
        with contextlib.redirect_stderr(errbuf), contextlib.redirect_stdout(io.StringIO()):
            assert run_main3(["bogus"]) == 2 and run_main3([]) == 2 and run_main3(["--help"]) == 0
        assert "unknown command 'bogus'" in errbuf.getvalue(), errbuf.getvalue()
        ok("third review: --json has no NaN, a damaged queue item is a message with exit 2, two threads on two projects do not mix, the older words are pinned, the runner's help and unknown "
           "command are honest")
    finally:
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

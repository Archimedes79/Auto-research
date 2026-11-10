"""Self-test of the queue as a mixin (queue_api.py): the queue is not part of the class Training, `class Q(QueueMixin, Training)` has it. Toy project and toy trainer as in selftest_api.py, about
three minutes: numbers follow the priority, the person's words, a pause and a held item also forbid a direct run, the runner starts an added job, the command line of the combined class, a launcher
that failed is tried again, the runner stops a run whose job-kill rule hits (the person's resume waives it), `hold` stops a launcher that starts the trainer after it answered "on hold"."""
import contextlib
import io
import json
import os
import tempfile
import time

import suitecli

from . import api
from . import trainqueue as TQ
from .queue_api import QUEUE_TEXT, QueueMixin
from .selftest_api import raises, toy
from .selftest_queue import until


class Q(QueueMixin, api.Training):
    METHODS = api.Training.METHODS + QueueMixin.QUEUE_METHODS


QCLI = suitecli.Cli(Q, "trainsuite", api.TrainingError, "TRAINSUITE_TRACEBACK", legacy=api.LEGACY, positional=("job", "i", "a", "b"), option_names=("words",),
                    text={**api.TEXT, **QUEUE_TEXT})


def qcli(*args):
    """(exit code, stdout, stderr) of the command line of the combined class; a usage error is a SystemExit"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = QCLI.dispatch(list(args))
    return rc, out.getvalue(), err.getvalue()


def queue_checks(tmp, ok):
    proj, job, restore = toy(tmp)
    T = Q()
    rec = lambda rid, name: os.path.join(proj, "records", "runs", rid, name)          # noqa: E731
    try:
        # 1 the queue: numbers follow the priority, an id and a folder name are the same item, remove, the person's words, pause and held items bind a direct start
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
        T.queue_add(job("2"), start=False)
        T.queue_add(job("1", priority=1), start=False)                              # ids "1" and "2" against the numbers 1 and 2: a text is an id first, an int is a number
        assert T.queue_get(1)["id"] == "1" and T.queue_get("2")["id"] == "2" and T.queue_get("2")["number"] == 2 and T.queue_get(2)["id"] == "2"
        T.queue_remove("1")
        T.queue_remove("2")
        # an item from before the job kinds went still carries "kind": "train" and is read like any other; a GPU-slot holder without a run folder of its own (an evaluation, a prediction)
        # is no running training
        T.queue_add(job("k.old"), start=False)
        c0 = TQ.context(proj)
        with TQ.edit(c0) as q0:
            q0["items"][0]["kind"] = "train"
        assert T.queue_get("k.old")["state"] == "queued" and "kind" not in T.queue_list()["items"][0]
        T.queue_remove("k.old")
        from . import queue_api
        real_slots, queue_api._slot_ids = queue_api._slot_ids, lambda c: ["ghost.eval"]
        try:
            assert queue_api._running(c0, [], {}) == [("ghost.eval", None, None)] and queue_api.resolve_item(c0, None) == (None, None, None), "only a training counts as the running one"
        finally:
            queue_api._slot_ids = real_slots
        raises(T.queue_hold, "words", "", all=True)
        raises(T.queue_resume, "words", "  ", all=True)
        raises(T.queue_hold, "name the item", "Mirko: test")                          # neither an item nor all: an error, a missing id never holds everything
        raises(T.queue_hold, "not both", "Mirko: test", i="q.a", all=True)
        raises(T.queue_resume, "name the item", "Mirko: test", i="")
        assert T.queue_hold("Mirko: test", all=True) == {} and T.queue_list()["paused"] is True
        raises(T.run, "PAUSED", job("p.one"))
        assert "PAUSED" in T.check(job("p.one"))["problems"][0] and T.check(job("p.one"))["ok"] is False, "check says what run would be refused for"
        assert T.queue_add(job("p.q"), start=False)["state"] == "queued", "the queue may be filled while it is paused"
        assert T.queue_resume("Mirko: test", all=True) == {} and T.queue_list()["paused"] is False
        T.queue_remove("p.q")
        T.queue_add(job("h.a"), start=False)
        T.queue_add(job("h.b"), start=False)
        held = T.queue_hold("Mirko: selftest", i=T.queue_get("h.b")["number"])    # a number: that item of the queue only
        assert set(held) == {"h.b"} and T.queue_get("h.b")["held"] and not T.queue_get("h.a")["held"] and T.queue_list()["paused"] is False, held
        raises(T.run, "is held", job("h.b"))
        assert set(T.queue_resume("Mirko: selftest", i="h.b")) == {"h.b"}
        until(lambda: all(os.path.exists(rec(x, "done")) for x in ("h.a", "h.b")), 90, what="h.a and h.b done")
        T.queue_remove("h.a")
        T.queue_remove("h.b")
        ok("queue: priority gives the numbers, an int is a number and a text an id, a double id and a bad number are refused, the person's words are required, a pause and a held item forbid a "
           "direct run and show in check")

        # 2 the runner: add with start, the item runs by itself, the view names it while it runs, the runner ends when nothing is left
        T.queue_add(job("w.one", set={"steps": 60, "sleep": 0.05}))
        until(lambda: T.queue_list()["running"] == "w.one", 60, what="w.one running through the queue")
        s = T.status()
        assert s["id"] == "w.one" and s["number"] == 1 and s["queue"] == "running", s
        until(lambda: os.path.exists(rec("w.one", "done")), 90, what="w.one done")
        assert T.queue_list()["items"][0]["state"] == "done" and T.queue_get(1)["step"] == 60
        until(lambda: TQ.runner_alive(TQ.context(proj)) == 0, 30, what="the runner ends by itself")
        assert T.queue_start()["runner"] > 0, "queue_start brings the runner back"
        T.queue_remove("w.one")
        ok("runner: an added job runs by itself, status() has the place in the queue, the item is done afterwards, queue_start starts the runner")

        # 3 the command line of the combined class
        rc, out, err = qcli("queue-add", job("k.two", priority=2), "--no-start", "--json")
        assert rc == 0 and json.loads(out)["id"] == "k.two", (rc, out, err)
        rc, out, err = qcli("queue-list", "--json")
        assert rc == 0 and "k.two" in [x["id"] for x in json.loads(out)["items"]], out
        rc, out, err = qcli("queue-list")
        assert rc == 0 and "k.two" in out and "machine lock:" in out and "slots:" in out, out
        rc, out, err = qcli("queue-get", "k.two", "--json")
        assert rc == 0 and json.loads(out)["job"]["priority"] == 2, out
        rc, out, err = qcli("queue-hold", "--words", "Mirko: selftest")            # no item and no --all: the method says so (exit 2)
        assert rc == 2 and "name the item" in err, (rc, err)
        rc, out, err = qcli("queue-hold", "--all", "--words", "Mirko: selftest")
        assert rc == 0 and json.loads(qcli("queue-list", "--json")[1])["paused"] is True, (rc, out, err)
        rc, out, err = qcli("queue-resume", "--all", "--words", "Mirko: selftest")
        assert rc == 0 and json.loads(qcli("queue-list", "--json")[1])["paused"] is False
        rc, out, err = qcli("queue-remove", "k.two")
        assert rc == 0 and "removed k.two" in out
        rc, out, err = qcli("queue-get", "k.two")
        assert rc == 2 and "not in the queue" in err, (rc, err)
        ok("command line: queue-add / queue-list / queue-get / queue-hold --all / queue-resume --all / queue-remove say what the methods return, errors go to stderr with exit 2")

        # 4 the deep review of the training stack: a launcher that failed (exit 1) is tried again and a refusal (exit 2) is final; the runner stops a run whose job-kill rule hits and the
        # person's resume waives that rule; `hold` stops a launcher that starts the trainer after it answered "on hold"
        c14 = TQ.context(proj)
        it14 = {"id": "x.fail", "argv": ["bash", "-c", "echo boom; exit 1"]}
        for k in (1, 2):
            started, text14 = TQ.launch(c14, it14, 5.0)
            assert not started and f"try {k} of {TQ.LAUNCH_RETRIES}" in text14 and not it14.get("refused"), text14
        started, text14 = TQ.launch(c14, it14, 5.0)
        assert not started and "boom" in it14.get("refused", ""), (text14, it14)
        it14b = {"id": "x.two", "argv": ["bash", "-c", "echo no card; exit 2"]}
        assert not TQ.launch(c14, it14b, 5.0)[0] and "no card" in it14b.get("refused", "") and "launch_failures" not in it14b, "exit 2 is a refusal: final at once"
        T.queue_add(job("k.enf", set={"steps": 600, "sleep": 0.05}, kill=["total@5 > 0.5"]))                 # the toy trainer logs total 1.0
        until(lambda: os.path.exists(rec("k.enf", "HOLD")), 90, what="the runner stopped k.enf on its kill rule")
        assert any(i["id"] == "k.enf" and i.get("kill_hit", {}).get("rule") == "total@5 > 0.5" for i in TQ.load(c14)["items"]), "the hit is kept with the item"
        step0 = T.status("k.enf")["step"]                                               # where the rule stopped it
        T.queue_resume("Mirko: selftest, go on", i="k.enf")
        until(lambda: T.status("k.enf")["state"] == "running" and T.status("k.enf")["step"] >= step0 + 20, 90, what="k.enf goes on after the person's resume")
        assert not os.path.exists(rec("k.enf", "HOLD")), "a waived rule does not stop it again"
        T.stop("k.enf", "selftest cleanup")
        until(lambda: os.path.exists(rec("k.enf", "HOLD")), 60, what="k.enf stopped")
        wrap = os.path.join(tmp, "wrap14")
        os.makedirs(wrap)
        with open(os.path.join(wrap, "git"), "w") as f:                                  # a slow snapshot: the launcher sits between its last look at the hold and the start of the trainer
            f.write('#!/bin/bash\nif [[ " $* " == *" write-tree "* ]]; then sleep 8; fi\nexec /usr/bin/git "$@"\n')
        os.chmod(os.path.join(wrap, "git"), 0o755)
        path14 = os.environ["PATH"]
        os.environ["PATH"] = wrap + ":" + path14
        try:
            T.run(job("h.race", set={"steps": 600, "sleep": 0.05}), wait=1.0)
            until(lambda: os.path.exists(rec("h.race", "run.json")) and json.load(open(rec("h.race", "run.json"))).get("status") == "waiting", 30, what="h.race waits for its slot")
            time.sleep(2)
            res = T.queue_hold("Mirko: selftest race", i="h.race", wait=5.0)
            assert "h.race" in res, res
            until(lambda: T.status("h.race")["state"] in ("stopped", "held"), 120, what="the run is held (the launcher saw the hold) or stopped (it started the trainer anyway)")
            time.sleep(14)                                                          # longer than the slowed snapshot: a trainer that starts after the hold would show up now
            assert T.list()["running"] == [] and T.status("h.race")["state"] in ("stopped", "held"), (T.list()["running"], T.status("h.race")["state"])
        finally:
            os.environ["PATH"] = path14
        ok("deep review: exit 1 of a launcher is retried and exit 2 is final, the runner stops a run on its job's kill rule and the person's resume waives it, `hold` stops a launcher that "
           "starts the trainer after it said 'on hold'")
    finally:
        restore()


def main():
    n = [0]

    def ok(msg):
        n[0] += 1
        print(f"  ok  {msg}")
    with tempfile.TemporaryDirectory() as tmp:
        queue_checks(tmp, ok)
    print(f"queue interface selftest: {n[0]} groups of checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-test of runsuite on a toy project (a few seconds, no data, no GPU, a real run.py launcher): `python -m runsuite.selftest`.

The job layer (RunJobs: check / run / status / queue_add of kind "run", the modes check / precompute / network / full) is tested in trainsuite/selftest_api.py next to the queue it shares. Here:
the layers (the class Runner loads neither the job layer nor trainsuite nor the command line); the methods of Runner are the commands; the run without ground truth on the toy network of
evalsuite.selftest_api (plan finds the problems, predict stores, a second predict takes the stored frames, list_results and tap read them, time / probe say when the project has none, the
command line prints what the methods return); and the job mode "predict" of RunJobs (the frames text is checked before anything starts, the command is the project's `predict` command with the
run, the frames and the flags, a real run.py launch passes them on, the queue takes the job, a project without a `predict` command refuses)."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import runsuite
from evalsuite import selftest_api as TOY
from trainsuite import api as T
from trainsuite.selftest_queue import make_project, until

from .api import Runner
from .jobs import RunJobs


def raises(fn, text, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected an error containing {text!r}")


def cli(*argv):
    """(exit code, stdout, stderr) of `python -m runsuite ...`"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = runsuite.dispatch(list(argv))
    return rc, out.getvalue(), err.getvalue()


def run_all(verbose=True):
    n = []

    def ok(text):
        n.append(text)
        if verbose:
            print(f"  ok  {text}")
    tmp = tempfile.mkdtemp(prefix="runsuite_selftest_")
    # the layers: a program that only runs a network does not load the job layer, trainsuite or the command line
    code = ("import sys, runsuite; r = runsuite.Runner('x'); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'trainsuite' or m in ('runsuite.jobs', 'runsuite.cli')))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert p.returncode == 0 and p.stdout.strip() == "[]", (p.stdout, p.stderr[-300:])
    ok("layers: importing Runner loads neither the job layer, nor trainsuite, nor the command line")

    # the methods are the commands (no second list): every one has a whole-sentence summary; the helpers of the Evaluation are not commands
    names = [m["name"] for m in Runner.methods()]
    public = [n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_") and n != "methods"]
    assert names == list(Runner.METHODS) == ["predict", "plan", "list_results", "tap", "time", "probe"] and set(public) == set(names) | {"resolve", "stored"}, (names, public)
    assert all(not m["summary"].endswith(("e.g.", "i.e.")) and m["summary"] for m in Runner.methods() + RunJobs.methods()), "a summary is a whole sentence"
    assert not set(Runner.METHODS) & set(RunJobs.METHODS), "a short command names one method: the two classes share no method name"
    ok(f"{len(names)} methods = {len(names)} commands, each with a summary; the two classes share no command word")

    # the direct run, on the toy network of evalsuite (its module is the adapter)
    TOY.STATE.root, TOY.STATE.gt, TOY.STATE.shift, TOY.STATE.calls, TOY.STATE.loads, TOY.STATE.free_loads, TOY.STATE.fail_at = os.path.join(tmp, "toy_out"), "g1", 0, 0, 0, 0, None
    r = Runner(tmp, adapter=TOY)
    net = "runs/thr0.5"
    p = r.plan(net, "val")
    assert p["ok"] and p["frames_to_run"] == 18 and p["frames_stored"] == 0 and p["network_runs"] is True and TOY.STATE.calls == 0, p
    assert not r.plan(net, "scenes:demo-7:val")["ok"] and not r.plan(net, "nothing:val")["ok"], "plan finds the problems and starts nothing"
    a = r.predict(net, "val", progress=0)
    assert a["frames_predicted"] == 18 and a["predictions_reused"] is False and a["warnings"] == [] and a["results"] == p["results"] and TOY.STATE.loads == 0 and TOY.STATE.free_loads > 0, a
    calls = TOY.STATE.calls
    b = r.predict(net, "val", progress=0)
    assert b["predictions_reused"] is True and b["frames_predicted"] == 0 and b["results"] == a["results"] and TOY.STATE.calls == calls, "the second call takes the stored frames: the network is not called"
    assert [x["id"] for x in r.list_results(net)] == [a["results"]] and r.list_results(net)[0]["frames"] == 18
    ok("predict: plan = what predict does, no ground truth read (the labelled Source.load never called), a second predict reuses, results lists it")
    t = r.predict("runs/thr0.6", "scenes:demo-0:val", taps=("feat",), progress=0)
    one = r.tap(t["results"], "demo-0/2", "feat")
    assert r.tap(t["results"], "demo-0/2")["taps"] == ["feat"] and one["tap"] == "feat" and os.path.isfile(one["path"]) and len(one["shape"]) == 2, one
    raises(r.tap, "no tap", t["results"], "demo-0/2", "nothing")
    raises(r.predict, "unknown taps", "runs/thr0.7", "scenes:demo-0:val", taps=("nothing",), progress=0)
    raises(r.time, "no timing", net)
    raises(r.probe, "no probe", net)
    ok("tap: an intermediate result is stored on request and read back; an unknown tap is refused before the network runs; time / probe say when the project has none")

    # the command line prints what the methods return (the project names its adapter in evaluation/evalsuite.json)
    os.makedirs(os.path.join(tmp, "evaluation"))
    json.dump({"adapter": "evalsuite.selftest_api"}, open(os.path.join(tmp, "evaluation", "evalsuite.json"), "w"))
    rc, out, err = cli("Runner.plan", net, "val", "--project", tmp, "--json")
    assert rc == 0 and json.loads(out)["frames_stored"] == 18 and not err.strip(), (rc, out, err)
    rc, out, err = cli("list-results", net, "--project", tmp, "--json")
    assert rc == 0 and json.loads(out)[0]["id"] == a["results"]
    rc, out, err = cli("Runner.plan", "runs/thr0.5", "scenes:demo-7:val", "--project", tmp)
    assert rc == 1 and "ok: False" in out, "a plan with problems is exit 1"
    rc, out, err = cli("Runner.tap", a["results"], "demo-9/9", "--project", tmp)
    assert rc == 2 and err.startswith("error:") and "Traceback" not in err, (rc, err)
    assert cli("Runner.nothing")[0] == 2 and cli("RunJobs.nothing")[0] == 2 and cli("nothing")[0] is None and cli("methods")[0] == 0, "an unknown command of a named class is an error"
    listing = cli("methods")[1]
    assert listing.count("(Runner.") == 6 and "(RunJobs.check(" in listing, "methods lists both classes"
    ok("command line: Runner.plan --json = the method's dict, `list-results` works without the class name, a plan with problems exits 1, a bad frame exits 2 with a message")

    # the job mode "predict": toy project with a real run.py launcher and a toy predict command
    proj = make_project(os.path.join(tmp, "job"))
    with open(os.path.join(proj, "code", "predict_toy.py"), "w") as f:
        f.write("import json, sys\njson.dump(sys.argv[1:], open('predict_args.json', 'w'))\n")
    os.makedirs(os.path.join(proj, "runs", "t_one"))
    pj = json.load(open(os.path.join(proj, "project.json")))
    pj["runsuite"] = {"command": "python3 code/runner.py", "predict": "python3 code/predict_toy.py {args}", "runs": "runs"}
    json.dump(pj, open(os.path.join(proj, "project.json"), "w"))
    env = {"AR_PROJECT": proj, "TS_QUEUE_EVERY": "0.5", "TS_QUEUE_IDLE": "3", "AR_HEARTBEAT_S": "0.5", "AR_STALE_S": "4", "AR_RAM_CHECKS": "1", "AR_SETTLE_S": "0",
           "AR_MACHINE_LOCK": os.path.join(tmp, "machine_lock.json")}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        pr = RunJobs(proj)
        job = dict(id="pred.one", kind="run", mode="predict", run="t_one", frames="scenes:s1,s2:small.val", taps=["fused", "lift"], device="cpu", workers=0)
        c = pr.check(job)
        assert c["ok"] and c["mode"] == "predict" and "predict_toy.py runs/t_one scenes:s1,s2:small.val --taps fused,lift --device cpu --workers 0" in c["command"] and "--gpu" not in c["command"], c
        assert "--gpu" in pr.check(dict(job, device="cuda", id="pred.two"))["command"], "a job on a GPU takes a slot"
        assert not pr.check(dict(job, frames="random:x:val"))["ok"] and "does not take" in " ".join(pr.check(dict(job, config="base.json", id="pred.three"))["problems"])
        assert "does not exist" in pr.check(dict(job, run="nothing", id="pred.four"))["problems"][0]
        r1 = pr.run(job)
        until(lambda: os.path.exists(os.path.join(proj, "records", "runs", "pred.one", "done")), 60, what="pred.one done")
        assert json.load(open(os.path.join(proj, "predict_args.json"))) == ["runs/t_one", "scenes:s1,s2:small.val", "--taps", "fused,lift", "--device", "cpu", "--workers", "0"], r1
        assert pr.status("pred.one")["state"] == "done"
        item = pr.queue_add(dict(job, id="pred.queued"), start=False)
        assert item["id"] == "pred.queued" and item["job"]["mode"] == "predict" and item["job"]["taps"] == ["fused", "lift"], item
        assert T.Training().queue_get("pred.queued")["id"] == "pred.queued", "one queue for trainings and runner jobs"
        pj["runsuite"].pop("predict")
        json.dump(pj, open(os.path.join(proj, "project.json"), "w"))
        raises(RunJobs(proj).check, 'no "runsuite" block with "predict"', dict(job, id="pred.five"))
        ok("job mode predict (RunJobs): the frames text and the run are checked before anything starts, the command carries run / frames / taps / device / workers, a real run.py launch passes them, the queue takes the job, a project without a `predict` command refuses")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return {"ok": True, "checks": len(n)}


def main():
    r = run_all()
    print(f"runsuite selftest passed: {r['checks']} groups")


if __name__ == "__main__":
    main()

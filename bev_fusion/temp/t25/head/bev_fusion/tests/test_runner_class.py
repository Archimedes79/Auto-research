"""runner/ (the class Runner and the command line `python -m runner` above it) on the toy network of autoresearch/Evaluation/selftest_api.py: a few seconds, no data, no GPU, no nuScenes.
Checked: (1) layering: `runner.runner` loads neither torch nor any suite besides Evaluation (the runner imports the evaluation's pieces; the evaluation reaches the runner only by the name in evaluation.json); (2) the methods of
Runner are the commands (no second list), each with a whole-sentence summary, `resolve` / `stored` are helpers and no commands; (3) `plan` finds the problems and runs nothing, `predict` stores
without reading a ground truth (the labelled Source.load is never called) and a second `predict` takes the stored frames; (4) `tap` reads an intermediate result back, an unknown tap is refused
before the network runs and plan says so too, a refused run stores nothing, the health of the memory (a warning that a shorter look keeps), `time` / `probe` say when the project has none; (5) the command line prints what the methods return: `--json` is the method's dict, a plan with problems exits 1, an
error exits 2 with a message and no traceback, a short word and `Runner.<word>` are the same command. The real network on real frames is tests/test_runner_real.py.
Run (from the bev_fusion folder): ~/miniconda3/envs/gcar/bin/python tests/test_runner_class.py"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from runner import __main__ as command_line  # noqa: E402
from runner import Runner  # noqa: E402

import suitecli  # noqa: E402
from Evaluation import selftest_api as TOY  # noqa: E402


def raises(fn, text, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected an error containing {text!r}")


def cli(*argv):
    """(exit code, stdout, stderr) of `python -m runner ...`"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = command_line.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def main():
    # (1) layering, in a process of its own
    code = ("import sys; sys.path.insert(0, %r); import runner.runner; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'Training', 'suitecli', 'nuscenes', 'cv2')))" % ROOT)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(ROOT))
    assert p.returncode == 0 and p.stdout.strip() == "[]", (p.stdout, p.stderr[-300:])

    # (2) the methods are the commands
    infos = suitecli.method_list(Runner)
    names = [m["name"] for m in infos]
    public = [n for n, f in vars(Runner).items() if callable(f) and not n.startswith("_")]
    assert names == list(Runner.METHODS) == ["predict", "plan", "list_results", "tap", "time", "probe"] and set(public) == set(names), (names, public)
    assert all(m["summary"] and not m["summary"].endswith(("e.g.", "i.e.")) for m in infos), "a summary is a whole sentence"
    assert all(n.split("_")[0] in {"predict", "plan", "list", "tap", "time", "probe"} for n in names)

    # (3) the run without ground truth, on the toy network (its module is the adapter)
    tmp = tempfile.mkdtemp(prefix="runner_class_test_")
    TOY.STATE.root, TOY.STATE.gt, TOY.STATE.shift, TOY.STATE.calls, TOY.STATE.loads, TOY.STATE.free_loads, TOY.STATE.fail_at = os.path.join(tmp, "toy_out"), "g1", 0, 0, 0, 0, None
    r = Runner(tmp, adapter=TOY)
    net = "runs/thr0.5"
    p = r.plan(net, "val")
    assert p["ok"] and p["frames_to_run"] == 18 and p["frames_stored"] == 0 and p["network_runs"] is True and TOY.STATE.calls == 0, p
    assert not r.plan(net, "scenes:demo-7:val")["ok"] and not r.plan(net, "nothing:val")["ok"], "plan finds the problems and starts nothing"
    a = r.predict(net, "val", progress=0)
    assert a["frames_predicted"] == 18 and a["predictions_reused"] is False and a["warnings"] == [] and a["results"] == p["results"], a
    assert TOY.STATE.loads == 0 and TOY.STATE.free_loads > 0, "no ground truth is read: only the free Source.load ran"
    calls = TOY.STATE.calls
    b = r.predict(net, "val", progress=0)
    assert b["predictions_reused"] is True and b["frames_predicted"] == 0 and b["results"] == a["results"] and TOY.STATE.calls == calls, "the second call takes the stored frames: the network is not called"
    assert [x["id"] for x in r.list_results(net)] == [a["results"]] and r.list_results(net)[0]["frames"] == 18
    print("  ok  plan = what predict does; no ground truth read; a second predict reuses; list_results shows it")

    # (4) taps and the things the project may not have
    t = r.predict("runs/thr0.6", "scenes:demo-0:val", taps=("feat",), progress=0)
    one = r.tap(t["results"], "demo-0/2", "feat")
    assert r.tap(t["results"], "demo-0/2")["taps"] == ["feat"] and one["tap"] == "feat" and os.path.isfile(one["path"]) and len(one["shape"]) == 2, one
    raises(r.tap, "no tap", t["results"], "demo-0/2", "nothing")
    calls = TOY.STATE.calls
    raises(r.predict, "unknown taps", "runs/thr0.7", "scenes:demo-0:val", taps=("nothing",), progress=0)
    assert TOY.STATE.calls == calls and r.list_results("runs/thr0.7") == [], "a refused run runs no frame and leaves no (empty) result set behind"
    bad = r.plan("runs/thr0.7", "scenes:demo-0:val", taps=("nothing",))
    assert not bad["ok"] and "unknown taps" in bad["problems"][0] and TOY.STATE.calls == calls, "plan finds what predict would refuse"
    raises(r.time, "no timing", net)
    raises(r.probe, "no probe", net)
    print("  ok  tap: stored on request and read back; an unknown tap is refused before the network runs, stores nothing and plan says so; time / probe say when the project has none")

    # (4b) the health of the memory: predict returns the warning, a shorter look at the same scene does not erase it, a memory that is never filled by the network is no warning but is said
    try:
        TOY.STATE.memory = "flat"
        w = r.predict("runs/thr0.2", "scenes:demo-1:val", progress=0)
        assert w["memory_checked"] is True and len(w["warnings"]) == 1 and "does not fill" in w["warnings"][0] and "demo-1" in w["warnings"][0], w
        look = r.predict("runs/thr0.2", "frames:demo-1/0,demo-1/1:val", taps=("feat",), progress=0)
        assert look["frames_predicted"] == 2 and look["warnings"] == w["warnings"], ("a run of two frames cannot tell whether the memory fills: it keeps the recorded warning", look)
        again = r.predict("runs/thr0.2", "scenes:demo-1:val", progress=0)
        assert again["predictions_reused"] is True and again["warnings"] == w["warnings"], again
        assert r.predict(net, "val", progress=0)["memory_checked"] is None, "nothing ran: nothing was checked"
    finally:
        TOY.STATE.memory = "ok"
    print("  ok  memory health: predict returns the warning of a memory that never fills; a shorter look at the scene keeps it; memory_checked says whether it could be judged")

    # (5) the command line (the project names its adapter in evaluation/evaluation.json)
    os.makedirs(os.path.join(tmp, "evaluation"))
    json.dump({"adapter": "Evaluation.selftest_api"}, open(os.path.join(tmp, "evaluation", "evaluation.json"), "w"))
    rc, out, err = cli("Runner.plan", net, "val", "--project", tmp, "--json")
    assert rc == 0 and json.loads(out)["frames_stored"] == 18 and not err.strip(), (rc, out, err)
    rc, out, err = cli("list-results", net, "--project", tmp, "--json")
    assert rc == 0 and json.loads(out)[0]["id"] == a["results"], (rc, out, err)
    assert cli("Runner.list-results", net, "--project", tmp)[0] == 0 and cli("list_results", net, "--project", tmp)[0] == 0, "short word, class word, underscore: one command"
    rc, out, err = cli("Runner.plan", net, "scenes:demo-7:val", "--project", tmp)
    assert rc == 1 and "ok: False" in out, "a plan with problems exits 1"
    rc, out, err = cli("Runner.tap", a["results"], "demo-9/9", "--project", tmp)
    assert rc == 2 and err.startswith("error:") and "Traceback" not in err, (rc, err)
    assert cli("Runner.nothing")[0] == 2 and cli("nothing")[0] == 2 and cli()[0] == 2 and cli("methods")[0] == 0 and cli("-h")[0] == 0
    listing = cli("methods")[1]
    assert listing.count("(Runner.") == 6 and "predict  (Runner.predict(network, frames='val', taps=()" in listing, listing[:300]
    print("  ok  command line: --json is the method's dict, a short word is the command, a plan with problems exits 1, an error exits 2 with a message, `methods` lists the six")
    shutil.rmtree(tmp, ignore_errors=True)                                  # (a failed run leaves its folder for a look)
    print(f"test_runner_class passed: layering, {len(names)} methods = {len(names)} commands, plan / predict / list_results / tap, the command line")


if __name__ == "__main__":
    main()

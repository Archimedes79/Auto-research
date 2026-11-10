"""Self-test of the jobs of GtGenerator (jobs.py): the table is a valid catalogue of the job runner, readable without numpy, its class has the arguments of the real commands, and
`JobRunner.check` accepts a GT job on a throw-away project and refuses a wrong one - nothing is generated. `python -m GtGenerator selftest --only jobs`; skipped (False) where
autoresearch/jobrunner is not there (a copy of GtGenerator alone)."""
import inspect
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AR = os.path.join(REPO, "autoresearch")


def _own_checks():
    from . import gt_api, jobs
    # the table: both commands, no card (a GT build never feeds a verdict), the interpreter is the script that pins nice 15 / one thread
    assert sorted(jobs.JOBS) == ["GtGenerator.generate_bev_gt", "GtGenerator.generate_lane_gt"], sorted(jobs.JOBS)
    for call, e in jobs.JOBS.items():
        assert e["suite"] == "GtGenerator.jobs" and call == f"{e['cls']}.{e['method']}" and e["how"] == "process" and not e.get("may_feed_verdict"), call
        assert e["python"] == jobs.NICE and os.access(jobs.NICE, os.X_OK) and not e["needs"].get("gpu"), (call, e["python"])
    # the job class has the arguments of the real commands (the signature the job runner checks a job against), except the generator options a job does not carry
    for name, real in (("generate_bev_gt", gt_api.generate_bev_gt), ("generate_lane_gt", gt_api.generate_lane_gt)):
        mine = [p for p in inspect.signature(getattr(jobs.GtGenerator, name)).parameters.values() if p.name != "self"]
        theirs = [p for p in inspect.signature(real).parameters.values() if p.kind != p.VAR_KEYWORD]
        expected = [(p.name, p.empty if (name, p.name) == ("generate_lane_gt", "rig") else p.default) for p in theirs]          # the lane job has no `rig="all"`: one job per system
        assert [(p.name, p.default) for p in mine] == expected, f"{name}: the job class drifted from gt_api.{name}"
    assert set(jobs.RIGS) == set(gt_api.SYSTEMS), (jobs.RIGS, gt_api.SYSTEMS)
    # the result is plain data with ok = complete
    class FakeKey:
        complete = False

        def __str__(self):
            return "fake | INCOMPLETE"

        def as_dict(self):
            return {"id": "abc", "complete": self.complete}
    try:
        jobs.GtGenerator().generate_lane_gt("~/data/nuscenes", "all")
        raise AssertionError("rig all must be refused: one job per system")
    except ValueError:
        pass
    r = jobs._plain(FakeKey())
    assert r["ok"] is False and r["id"] == "abc" and json.dumps(r), r
    FakeKey.complete = True
    assert jobs._plain(FakeKey())["ok"] is True


def _without_numpy():
    """the job runner reads `GtGenerator.JOBS` in its own interpreter (plain python3): in a fresh interpreter the table is there and numpy is not imported"""
    code = "import sys, GtGenerator; assert sorted(GtGenerator.JOBS) and 'numpy' not in sys.modules and 'GtGenerator.suite' not in sys.modules, sorted(m for m in sys.modules if m.startswith(('numpy', 'GtGenerator')))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
    assert r.returncode == 0, "GtGenerator.JOBS must be readable without importing numpy: " + r.stderr[-500:]


def _job_runner_check():
    """JobRunner.check on a throw-away project that lists GtGenerator in jobrunner.suites: a good job passes (the call, the process form, what it reads and writes), wrong ones are refused"""
    if not os.path.isdir(os.path.join(AR, "jobrunner")):
        return False
    sys.path.insert(0, AR)
    try:
        from jobrunner import JobRunner
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "code"))
            json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", runs="runs", schedule="schedule.csv", test_log="TEST.md", code_dirs=["code"], scratch="temp",
                           live_code=["code"], jobrunner=dict(suites=["GtGenerator"])), open(os.path.join(t, "project.json"), "w"))
            open(os.path.join(t, "TEST.md"), "w").write("| none |\n")
            data = os.path.join(t, "nuscenes")
            os.makedirs(data)
            jr = JobRunner(t)
            calls = [c["call"] for c in jr.catalogue()]
            assert "GtGenerator.generate_bev_gt" in calls and "GtGenerator.generate_lane_gt" in calls, calls
            good = jr.check({"id": "gt.bev.small", "call": "GtGenerator.generate_bev_gt", "args": {"dataroot": data, "set_name": "small"}})
            assert good["ok"] and good["how"] == "process" and not good["problems"], good
            assert any(x.startswith("path:") for x in good["reads"]) and good["writes"] == ["gt:front@small", "gt:lane.bev_fusion@small", "gt:masks@small"], good
            assert "jobrunner" in " ".join(good["command"]) and good["command"].count("GtGenerator.jobs") >= 1 and "run_nice.sh" in " ".join(good["command"]), good["command"]
            lane = jr.check({"id": "gt.lane.small", "call": "GtGenerator.generate_lane_gt", "args": {"dataroot": data, "rig": "bev_fusion", "set_name": "small"}})
            assert lane["ok"] and lane["writes"] == ["gt:lane.bev_fusion@small"], lane
            norig = jr.check({"id": "gt.lane.norig", "call": "GtGenerator.generate_lane_gt", "args": {"dataroot": data, "set_name": "small"}})
            assert not norig["ok"], "the lane job needs its rig"
            for bad_args, why in (({"dataroot": data, "set_nam": "small"}, "unknown argument"), ({"set_name": "small"}, "dataroot is required"), ({"dataroot": os.path.join(t, "missing")}, "the data folder must exist")):
                r = jr.check({"id": "gt.bad", "call": "GtGenerator.generate_bev_gt", "args": bad_args})
                assert not r["ok"] and r["problems"], (why, r)
            card = jr.check({"id": "gt.card", "call": "GtGenerator.generate_bev_gt", "args": {"dataroot": data}, "card": {"hyp": "h", "pred": "p", "rule": "r"}})
            assert not card["ok"], "a GT build never feeds a verdict: a card is refused"
    finally:
        sys.path.remove(AR)
    return True


def run(verbose=True):
    _own_checks()
    _without_numpy()
    checked = _job_runner_check()
    msg = ("jobs selftest ok: the table is a valid catalogue, the job class has the arguments of gt_api, ok = complete, GtGenerator.JOBS is readable without numpy"
           + ("; JobRunner.check accepts a GT job on a throw-away project (reads, writes, process form, run_nice.sh) and refuses a wrong argument, a missing dataroot or folder and a card" if checked else ""))
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    run()

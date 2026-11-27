"""The test of the research system: known answers of its parts, then the whole chain on the toy problem in a throw-away repository (nothing of
this checkout and no machine state is touched).

    python -m research.selftest [--quick]        (from autoresearch/, with the interpreter that has torch; --quick: the parts only, seconds)

The chain: new -> check -> start (snapshot, tag, workers) -> runs from the snapshot -> judge (verdict, ledger, commit) -> adopt (anchor) and the
refusals: a card changed after its start, an adoption whose anchor test or commit fails or whose run folder is gone (base config and
champions/ put back), an untrained control (INVALID), a kill rule (also on a later row, in a step's own folder), stop and resume, a
registration without a DONE verdict, a holdout read (also with the variables of a TEST run but no log row), an evaluation by hand, an edit of
a protected file or the state folder without and with a grant (also with a broken gates.py), and clean (a list, then exactly it under the
decided row that names its hash)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CORE)


def say(text: str) -> None:
    print(f"ok  {text}", flush=True)


def expect_error(fn, text: str, *a, **k):
    from research.paths import ResearchError
    try:
        fn(*a, **k)
    except (ResearchError, PermissionError, ValueError) as e:
        assert text in str(e), f"expected {text!r} in {e}"
        return str(e)
    raise AssertionError(f"no refusal ({text})")


# ------------------------------------------------------------------------------------------------ parts
def parts(tmp: str) -> None:
    from research import gates, hook, judge, spec, worker
    from research.paths import child_env, machine_file, parse_run, run_id
    p = json.load(open(os.path.join(REPO, "problems", "toy", "problem.json"), encoding="utf-8"))
    assert spec.problem_errors(p, "toy") == []
    bad = dict(p, scroe=1)
    assert any("unknown key 'scroe'" in e for e in spec.problem_errors(bad, "toy"))
    stage = lambda d: dict(p, steps=dict(p["steps"], stage={"run": ["{python}", "x.py"], "dir": d}))      # noqa: E731
    assert spec.problem_errors(stage("stage1"), "toy") == [] and any("subfolder" in e for e in spec.problem_errors(stage("../x"), "toy"))
    base = json.load(open(os.path.join(REPO, "problems", "toy", "training", "configs", "base.json"), encoding="utf-8"))
    card = {"id": "E1", "title": "t", "question": "q", "hypothesis": "h", "prediction": 0.02, "control": "c", "seeds": [0, 1],
            "arms": {"c": {"set": {"width": 4}}, "a": {"set": {"widht": 8}}}}
    errs = spec.card_errors(card, p, base)
    assert any("'widht' is not a setting" in e for e in errs) and len(errs) == 1, errs
    assert any("holdout" in e for e in spec.card_errors(dict(card, split="test", arms={"c": {}, "a": {}}), p, base))
    assert any("control" in e for e in spec.card_errors(dict(card, control="x", arms={"c": {}, "a": {}}), p, base))
    measure = {**{k: v for k, v in card.items() if k not in ("control", "prediction")}, "kind": "measure", "arms": {"m": {}}}
    assert spec.card_errors(measure, p, base) == []
    written = dict(card, hypothesis=spec.PLACEHOLDERS["hypothesis"], arms={"c": {"set": {"width": 4}}, "a": {"set": {"width": 8}}})
    assert spec.card_errors(written, p, base) == ["card: 'hypothesis' is still the template's text"], spec.card_errors(written, p, base)
    same = dict(card, arms={"c": {}, "a": {"set": {"width": base["width"]}}})
    assert any("same settings as the control" in e for e in spec.card_errors(same, p, base))
    template = open(os.path.join(CORE, "templates", "experiment.md"), encoding="utf-8").read()
    assert all(v in template for v in spec.PLACEHOLDERS.values()), "the placeholders are the template's texts"
    cfg = spec.compose(base, {"arms": {"a": {"set": {"width": 12}, "needs": {"data_root": "E3.b"}}}}, "a", 1, "E1.a_s1")
    assert cfg["width"] == 12 and cfg["seed"] == 1 and cfg["name"] == "E1.a_s1" and cfg["out_root"] == "{runs}" and cfg["data_root"] == "{runs}/E3.b_s1"
    assert parse_run(run_id("E12", "b2", 3)) == ("E12", "b2", 3)
    expect_error(run_id, "an arm is", "E1", "B", 0)
    # the standard rule
    assert judge.standard(0.05, 0.02, 0.08, 0.01)[0] == "KEEP"
    assert judge.standard(0.005, 0.001, 0.009, 0.01)[0] == "TIE"
    assert judge.standard(-0.05, -0.08, -0.02, 0.01)[0] == "CUT"
    assert judge.standard(-0.002, -0.005, -0.0001, 0.01)[0] == "TIE"           # significant but inside the band: a tie
    assert judge.standard(0.05, -0.01, 0.1, 0.01)[0] == "INCONCLUSIVE"
    assert judge.standard(float("nan"), 0.0, 0.1, 0.01)[0] == "INCONCLUSIVE"
    # lower is better: a variant 0.05 lower than the control is a gain of +0.05; guards in both directions, an undefined number breaks its guard
    assert judge.gain_interval({"diff": -0.05, "lo": -0.08, "hi": -0.02}, -1.0) == (0.05, 0.02, 0.08)
    assert judge.standard(*judge.gain_interval({"diff": -0.05, "lo": -0.08, "hi": -0.02}, -1.0), 0.01)[0] == "KEEP"
    nums = {"m.up": {"control": 0.80, "variant": 0.78, "better": "high"}, "m.down": {"control": 10.0, "variant": 11.5, "better": "low"},
            "m.nan": {"control": 0.5, "variant": float("nan"), "better": "high"}}
    assert judge.guards_broken([{"metric": "m.up", "max_drop": 0.01}], nums) and not judge.guards_broken([{"metric": "m.up", "max_drop": 0.05}], nums)
    assert judge.guards_broken([{"metric": "m.down", "max_drop": 1.0}], nums) and not judge.guards_broken([{"metric": "m.down", "max_drop": 2.0}], nums)
    assert judge.guards_broken([{"metric": "m.nan", "min": 0.0}], nums)
    # the control: an undefined seed fails the floor and the spread
    assert judge.control_problems({"0": 0.50, "1": 0.52}, 0.3, 0.05, 1.0, "s") == []
    assert len(judge.control_problems({"0": float("nan"), "1": 0.5}, 0.3, 0.05, 1.0, "s")) == 2
    assert judge.control_problems({"0": 0.5, "1": 0.6}, None, 0.05, 1.0, "s") and judge.control_problems({"0": 0.2}, 0.3, None, 1.0, "s")
    # kill rules on a growing log: every row from after_step on (a later row fires), NaN fires, rows that are no step rows are skipped
    log = os.path.join(tmp, "log.jsonl")
    k = worker.Kill([{"metric": "total", "after_step": 5, "above": 0.5}], log)
    rows = [{"step": 2, "total": 0.9}, {"step": 10, "total": 0.1}, [1, 2], {"step": 15, "total": "n/a"}, {"step": "x", "total": 0.9}]
    with open(log, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows) + '{"step": 20, "tot')
    assert k.fired() == ""
    with open(log, "a") as f:
        f.write('al": 0.7}\n')
    assert "= 0.7 at step 20 is above 0.5" in k.fired()
    with open(log, "w") as f:
        f.write(json.dumps({"step": 1, "total": 0.0}) + "\n" + json.dumps({"step": 2, "total": float("nan")}) + "\n")
    assert "not a number at step 2" in worker.Kill([{"metric": "total", "after_step": 0, "below": -1.0}], log).fired()
    # a run that another needs: a stop request comes first, a stopped needed run stops this one (resumable), a failed one fails it
    assert worker.needed("E1.c_s0", "running", False) is None and worker.needed("E1.c_s0", "ok", True) == ""
    assert worker.needed("E1.c_s0", "failed", True).startswith("stopped") and worker.needed("E1.c_s0", "stopped", False).startswith("stopped")
    assert worker.needed("E1.c_s0", "failed", False) == "the needed run E1.c_s0 is failed"
    # decisions and the TEST log
    root = os.path.join(tmp, "repo_g")
    os.makedirs(os.path.join(root, "problems", "x"))
    json.dump(p | {"name": "x", "tests": ["hang.py"]}, open(os.path.join(root, "problems", "x", "problem.json"), "w"))
    open(os.path.join(root, "DECISIONS.md"), "w").write("# D\n## Decided\n| date | decision |\n|---|---|\n| 2026-11-27 | D7: ok |\n\n## Open\n| D8 | asked |\n")
    assert gates.decided("x", root) == {"D7"} and gates.decided_rows("x", root)["D7"] == "| 2026-11-27 | D7: ok |"
    log_path = os.path.join(root, "problems", "x", "TEST_LOG.md")
    open(log_path, "w").write("# TEST\n## Runs\n| date | run | decision | note |\n|---|---|---|---|\n| 2026-11-27 | E1.a_s0 | D8 | asked |\n| 2026-11-27 | E1.c_s0 | D7 | once |\n")
    assert "not a decided" in gates.test_refusal("x", "E1.a_s0", root)
    assert gates.test_refusal("x", "E1.c_s0", root) == ""
    assert "not named" in gates.test_refusal("x", "E1.b_s0", root)
    assert gates.is_protected("problems/x/problem.json", root) and gates.is_protected("problems/x/evaluation/adapter.py", root)
    assert gates.is_protected("champions/x/E1.a/network.json", root) and not gates.is_protected("problems/x/net/model/net.py", root)
    os.makedirs(os.path.join(root, "problems", "y"))                  # a problem whose list does not name its TEST log
    json.dump({k: v for k, v in p.items() if k != "protected"} | {"name": "y"}, open(os.path.join(root, "problems", "y", "problem.json"), "w"))
    assert gates.is_protected("problems/y/TEST_LOG.md", root) and gates.is_protected("problems/x/test_log.md", root)
    assert gates.is_protected("autoresearch/Evaluation/defaults.py", root) and not gates.is_protected("problems/x/DECISIONS.md", root)
    if os.name != "nt":
        assert hook.to_local("\\\\wsl.localhost\\Ubuntu\\home\\u\\.research\\state\\grants.json") == "/home/u/.research/state/grants.json"
    assert hook.strip_stream("C:\\x\\TEST_LOG.md::$DATA") == "C:/x/TEST_LOG.md" and hook.strip_stream("a/TEST_LOG.md:s") == "a/TEST_LOG.md"
    assert hook.strip_stream("C:/x/y.md") == "C:/x/y.md"
    assert {pat.lower() for pat in gates.ROOT_PROTECTED} <= set(hook.STATIC_PROTECTED), "the hook's static list (a broken gate module) covers the root list"
    # a step's environment: nothing of the caller's trees, the machine file and its own values; the rest (PATH) kept
    saved = dict(os.environ)
    try:
        os.environ.update(BFLC_ROOT="/elsewhere", RESEARCH_EVAL_OUT="/elsewhere", RESEARCH_RUN="E9.z_s0", GIT_DIR="/elsewhere/.git")
        e = child_env(RESEARCH_RUN="E1.a_s0")
        assert not {"BFLC_ROOT", "RESEARCH_EVAL_OUT", "GIT_DIR"} & set(e) and e["RESEARCH_RUN"] == "E1.a_s0" and e["RESEARCH_MACHINE"] == machine_file()
        assert e.get("PATH") == os.environ.get("PATH")
        # clean: a card names runs by from / needs and by a {runs}/<run> path in its settings
        from research import upkeep
        machine_p = os.path.join(tmp, "machine_p", "machine.json")
        os.makedirs(os.path.dirname(machine_p))
        json.dump({"data": os.path.join(tmp, "data_p"), "runs": os.path.join(tmp, "runs_p"), "python": {"gcar": sys.executable}, "gpu_slots": 0}, open(machine_p, "w"))
        os.environ.update(RESEARCH_REPO=root, RESEARCH_MACHINE=machine_p)
        os.makedirs(os.path.join(root, "problems", "x", "experiments", "E001_a"))
        write_card(os.path.join(root, "problems", "x", "experiments"), "E001_a",
                   {"id": "E1", "title": "t", "arms": {"r": {"from": "E2.c"}, "w": {"set": {"init": "{runs}/E3.b_s1"}}}})
        refs = upkeep.referenced("x")
        assert "names E3.b_s1" in upkeep.why_kept("x", "E3.b_s1", [], refs) and "names E2.c" in upkeep.why_kept("x", "E2.c_s0", [], refs), refs
        assert upkeep.kept_runs("x") == [], "nothing registered yet: nothing kept, no refusal"
        from research import index
        assert index.current("x"), "no run recorded yet: no ledger needed"
        os.makedirs(os.path.join(root, "champions", "x", "E1.a"))
        open(os.path.join(root, "champions", "x", "E1.a", "network.json"), "w").write("{}")
        expect_error(upkeep.kept_runs, "cannot be read", "x")                       # a network without champions.json
        from research import cli
        assert cli._grant_path("problems/x/problem.json") == "problems/x/problem.json" and cli._grant_path("champions") == "champions/"
        assert cli._grant_path(os.path.join(root, "problems", "x", "TEST_LOG.md")) == "problems/x/TEST_LOG.md"
        expect_error(cli._grant_path, "not a path inside", "../elsewhere.md")
        # research tests: a test past its timeout is killed with everything it started
        with open(os.path.join(root, "problems", "x", "hang.py"), "w") as f:
            f.write("import subprocess, time\nopen('child.pid', 'w').write(str(subprocess.Popen(['sleep', '60']).pid))\ntime.sleep(60)\n")
        t = upkeep.tests("x", timeout=2)
        child = int(open(os.path.join(root, "problems", "x", "child.pid")).read())
        assert not t["passed"] and "timed out" in t["tests"][0]["last"] and ended(child), (t, child)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    # a card may only gain seeds after its start; nested settings merge key by key; results scored differently are not comparable
    assert spec.same_rule(card, dict(card, seeds=[0, 1, 2])) and not spec.same_rule(card, dict(card, seeds=[1, 0])) and not spec.same_rule(card, dict(card, prediction=0.1))
    assert spec.merged({"m": {"a": 1, "b": 2}, "c": 3}, {"m": {"b": 5}}) == {"m": {"a": 1, "b": 5}, "c": 3}
    assert spec.unknown_settings({"m": {"x": 1}}, {"m": {"a": 1}}) == ["m.x"]
    r = {"split": "val", "frames": 160, "evaluation": {"gt": "g1", "evaluator": "e1", "metrics": ["seg"]}}
    assert judge.incomparable([{"result": r}, {"result": r}]) == []
    assert any("ground truth" in x for x in judge.incomparable([{"result": r}, {"result": dict(r, evaluation=dict(r["evaluation"], gt="g2"))}]))
    signed = lambda s: {"run": f"E1.a_{s}", "result": dict(r, evaluation=dict(r["evaluation"], gt_signature=s))}      # noqa: E731
    assert judge.incomparable([signed("s1"), signed("s1")]) == [] and any("signature" in x for x in judge.incomparable([signed("s1"), signed("s2")]))
    assert judge.incomparable([signed("s1"), {"result": r}]) == [] and judge.unsigned([signed("s1"), {"run": "E1.c_s0", "result": r}]) == ["E1.c_s0"]
    say("parts: schemas and typos, compose, run ids, the standard rule, the control check, kill rules, a needed run, decisions, the TEST log, "
        "protected paths, stream names, a step's environment, the references clean keeps, a test past its timeout")


def ended(pid: int, seconds: float = 5.0) -> bool:
    """has the process ended (gone or a zombie) within some seconds?"""
    end = time.time() + seconds
    while time.time() < end:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                if f.read().rsplit(")", 1)[-1].split()[0] in ("Z", "X"):
                    return True
        except FileNotFoundError:
            return True
        time.sleep(0.1)
    return False


# ------------------------------------------------------------------------------------------------ the chain
def sh(cmd, cwd, env, ok=(0,)):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode in ok, f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r


def research(args, env, ok=(0,)):
    return sh([sys.executable, "-m", "research", "--json", *args], env["RESEARCH_TMP_CORE"], env, ok)


STAGE_PROBE = '''"""selftest: a step that writes its log into <run>/stage1 (problem.json step "dir") and stops when CONTROL appears there"""
import json, os, sys, time
d = os.path.join(sys.argv[1], "stage1")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "log.jsonl"), "w") as f:
    f.write("".join(json.dumps(r) + "\\n" for r in ({"step": 10, "total": 0.1}, [1, 2], {"step": 15, "total": "n/a"}, {"step": 20, "total": 0.9})))
end = time.time() + 60
while time.time() < end:
    if os.path.exists(os.path.join(d, "CONTROL")):
        sys.exit(75)
    time.sleep(0.2)
'''


def make_repo(tmp: str) -> str:
    root = os.path.join(tmp, "repo")
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "temp", "eval_out", "state", "history")
    for d in ("research", "Evaluation", "trainkit", "templates"):
        shutil.copytree(os.path.join(CORE, d), os.path.join(root, "autoresearch", d), ignore=skip)
    toy = os.path.join(root, "problems", "toy")
    shutil.copytree(os.path.join(REPO, "problems", "toy"), toy, ignore=shutil.ignore_patterns("__pycache__", "experiments", "ledger.csv"))
    prob = json.load(open(os.path.join(toy, "problem.json"), encoding="utf-8"))
    prob["steps"]["stage"] = {"run": ["{python}", "stage_probe.py", "{run_dir}"], "dir": "stage1", "note": "selftest: a step with its own folder"}
    json.dump(prob, open(os.path.join(toy, "problem.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(toy, "stage_probe.py"), "w", encoding="utf-8") as f:
        f.write(STAGE_PROBE)
    with open(os.path.join(root, "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write("# Decisions\n\n## Decided\n| date | decision |\n|---|---|\n| 2026-11-27 | D1: the selftest's owner says yes |\n\n## Open\n| id | question |\n|---|---|\n")
    with open(os.path.join(root, "problems", "toy", "TEST_LOG.md"), "w", encoding="utf-8") as f:
        f.write("# TEST log\n\n## Runs\n| date | run | decision | note |\n|---|---|---|---|\n")
    with open(os.path.join(root, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.name", "research selftest"], ["git", "config", "user.email", "selftest@example.invalid"],
                ["git", "add", "-A"], ["git", "commit", "-qm", "selftest repository"]):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True)
    return root


def card_text(card: dict) -> str:
    return f"# {card['id']} {card['title']}\n\n```json card\n{json.dumps(card, indent=1)}\n```\n\n## Conclusion\n"


def write_card(root: str, folder: str, card: dict) -> None:
    with open(os.path.join(root, folder, "experiment.md"), "w", encoding="utf-8") as f:
        f.write(card_text(card))


def wait(root: str, folder: str, rids, states=("ok", "failed", "killed", "stopped"), seconds=900) -> dict:
    end = time.time() + seconds
    got = {}
    while time.time() < end:
        for rid in rids:
            try:
                got[rid] = json.load(open(os.path.join(root, folder, "runs", rid, "run.json"), encoding="utf-8"))["status"]
            except (OSError, ValueError, KeyError):
                got[rid] = "none"
        if all(s in states for s in got.values()):
            return got
        time.sleep(1)
    raise AssertionError(f"runs did not end in {seconds} s: {got}")


def run_trace(tmp: str, root: str, folder: str, rid: str) -> str:
    """what a failed run left: its record and the tails of its logs (for an assertion message)"""
    out = []
    for path in [os.path.join(root, folder, "runs", rid, "run.json")] + sorted(glob.glob(os.path.join(tmp, "runs", "toy", rid, "*.log"))):
        try:
            out.append(f"--- {os.path.basename(path)}\n" + open(path, encoding="utf-8", errors="replace").read()[-1500:])
        except OSError as e:
            out.append(f"--- {path}: {e}")
    return "\n".join(out)


def chain(tmp: str) -> None:
    root = make_repo(tmp)
    machine = os.path.join(tmp, "machine", "machine.json")
    os.makedirs(os.path.dirname(machine))
    json.dump({"data": os.path.join(tmp, "data"), "runs": os.path.join(tmp, "runs"), "python": {"gcar": sys.executable}, "gpu_slots": 1}, open(machine, "w"))
    core = os.path.join(root, "autoresearch")
    env = dict(os.environ, RESEARCH_MACHINE=machine, PYTHONPATH=core, RESEARCH_TMP_CORE=core)
    for k in ("RESEARCH_REPO", "RESEARCH_RUN", "RESEARCH_RECORD", "RESEARCH_HOLDOUT_RUN", "RESEARCH_RUNS", "RESEARCH_DATA"):
        env.pop(k, None)
    quick = {"steps": 60, "eval_every": 60, "save_every": 30, "log_every": 10, "threads": 1}
    # E1: two arms, two seeds
    e1 = json.loads(research(["new", "toy", "width", "width of the network"], env).stdout)["folder"]
    assert e1 == "problems/toy/experiments/E001_width", e1
    blank = json.loads(research(["check", "toy", "E1"], env, ok=(1,)).stdout)["errors"]     # the unwritten card: check exits 1
    assert any("template's text" in e for e in blank) and any("same settings" in e for e in blank), blank
    assert "a typo" not in blank and "no problem" in research(["hold", "toyy", "a typo"], env, ok=(1,)).stderr
    card = {"id": "E1", "title": "width", "question": "does a wider network find the disc better?", "hypothesis": "more channels, better masks",
            "prediction": 0.05, "control": "c", "seeds": [0, 1],
            "arms": {"c": {"set": {**quick, "steps": 300, "width": 8}}, "a": {"set": {**quick, "steps": 300, "width": 12}}},
            "rule": {"boot": 300}, "kill": [{"metric": "total", "after_step": 10, "above": 100.0}]}
    write_card(root, e1, card)
    assert json.loads(research(["check", "toy", "E1"], env).stdout)["errors"] == []
    with open(os.path.join(root, "notes.md"), "w", encoding="utf-8") as f:          # commit: a path named from the top, run from autoresearch/
        f.write("a note\n")
    assert json.loads(research(["commit", "notes.md", "-m", "a note"], env).stdout)["commit"] != "nothing changed"
    assert subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True).stdout.strip() == "a note"
    typo = research(["commit", "no/such.md", "-m", "a typo"], env, ok=(1,))
    assert "no such path" in typo.stdout + typo.stderr, typo
    net_py = os.path.join(root, "problems", "toy", "net", "model", "net.py")
    keep = open(net_py, encoding="utf-8").read()
    open(net_py, "a", encoding="utf-8").write("# a half-done edit of another chat\n")
    dirty = json.loads(research(["check", "toy", "E1"], env, ok=(1,)).stdout)["errors"]
    assert any("not committed" in e and "problems/toy/net/model/net.py" in e for e in dirty), dirty
    assert "not committed" in research(["start", "toy", "E1"], env, ok=(1,)).stderr
    open(net_py, "w", encoding="utf-8").write(keep)
    started = json.loads(research(["start", "toy", "E1"], env).stdout)
    assert sorted(started["started"]) == ["E1.a_s0", "E1.a_s1", "E1.c_s0", "E1.c_s1"] and started["tag"] == "toy/E1", started
    snap = subprocess.run(["git", "rev-parse", "toy/E1^{commit}"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert snap and subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() != snap
    st = wait(root, e1, started["started"])
    assert set(st.values()) == {"ok"}, (st, run_trace(tmp, root, e1, "E1.c_s0"))
    res = json.load(open(os.path.join(root, e1, "runs", "E1.a_s0", "result.json")))
    assert res["identity"]["commit"] == snap and res["split"] == "val" and res["frames"] == 160 and "seg.iou" in res["numbers"], res
    say(f"start: one snapshot {started['tag']}, four runs from it, results (E1.a_s0 IoU {res['score']['value']:.3f})")
    # a second start of the same experiment starts nothing and keeps the tag
    again = json.loads(research(["start", "toy", "E1"], env).stdout)
    assert again["started"] == [] and len(again["skipped"]) == 4 and again["commit"] == snap[:12]
    # the card may not change after its start
    text = open(os.path.join(root, e1, "experiment.md")).read()
    write_card(root, e1, dict(card, rule={"boot": 300, "min_effect": 0.5}))
    assert any("changed since its snapshot" in e for e in json.loads(research(["check", "toy", "E1"], env, ok=(1,)).stdout)["errors"])
    assert "changed since its snapshot" in research(["judge", "toy", "E1"], env, ok=(1,)).stderr
    assert "no run E1.zz_s0" in research(["stop", "toy", "E1.zz_s0"], env, ok=(1,)).stderr
    assert "numbers, comma separated" in research(["start", "toy", "E1", "--seeds", "abc"], env, ok=(1,)).stderr
    open(os.path.join(root, e1, "experiment.md"), "w").write(text + "\nThe conclusion may be written; the card block may not change.\n")
    v = json.loads(research(["judge", "toy", "E1", "--write", "--boot", "300"], env).stdout)
    arm = v["arms"]["a"]
    assert arm["verdict"] in ("KEEP", "TIE", "CUT", "INCONCLUSIVE") and len(arm["interval"]) == 2 and v["records_commit"] and v["snapshot"] == snap, v
    log = subprocess.run(["git", "log", "-1", "--name-only", "--format=%s"], cwd=root, capture_output=True, text=True).stdout
    assert "verdict" in log and "problems/toy/ledger.csv" in log and "verdict.json" in log, log
    assert json.loads(research(["index", "toy", "--check"], env).stdout)["current"]
    say(f"judge: {arm['verdict']} (gain {arm['gain']:+.3f} [{arm['interval'][0]:+.3f}, {arm['interval'][1]:+.3f}]), verdict.json, ledger, one commit")
    # adopt (by hand when it was no KEEP): the anchor evaluates again with the working tree's code and must agree bit for bit; an anchor test
    # that dies (here: a record is missing) puts the base config back
    by_hand = [] if arm["verdict"] == "KEEP" else ["--by-hand", "the selftest adopts whatever won"]
    base_json = os.path.join(root, "problems", "toy", "training", "configs", "base.json")
    before = open(base_json, encoding="utf-8").read()
    rec_a0 = os.path.join(root, e1, "runs", "E1.a_s0", "result.json")
    os.replace(rec_a0, rec_a0 + ".away")
    died = research(["adopt", "toy", "E1", "a", "--decision", "D1", *by_hand], env, ok=(1,))
    os.replace(rec_a0 + ".away", rec_a0)
    assert "anchor test of E1.a_s0 did not run" in died.stderr and open(base_json, encoding="utf-8").read() == before, died.stderr[-2000:]
    run_a1 = os.path.join(tmp, "runs", "toy", "E1.a_s1")
    os.replace(run_a1, run_a1 + ".away")                             # a seed's run folder is gone: refused before anything is written
    assert "run folders of E1.a_s1 are gone" in research(["adopt", "toy", "E1", "a", "--decision", "D1", *by_hand], env, ok=(1,)).stderr
    os.replace(run_a1 + ".away", run_a1)
    branch = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=root, check=True)       # the anchor test passes, the commit fails: all put back
    late = research(["adopt", "toy", "E1", "a", "--decision", "D1", *by_hand], env, ok=(1,))
    subprocess.run(["git", "checkout", "-q", branch], cwd=root, check=True)
    assert "detached" in late.stderr and open(base_json, encoding="utf-8").read() == before and not os.path.exists(os.path.join(root, "champions", "toy")), late.stderr[-2000:]
    ad = json.loads(research(["adopt", "toy", "E1", "a", "--decision", "D1", *by_hand], env).stdout)
    champ = json.load(open(os.path.join(root, "champions", "toy", "champions.json")))
    assert ad["champion"] == "E1.a" and champ["champion"] == "E1.a" and ad["commit"]
    assert json.load(open(os.path.join(root, "problems", "toy", "training", "configs", "base.json")))["width"] == 12
    expect = research(["adopt", "toy", "E1", "a", "--decision", "D2"], env, ok=(1,)).stderr
    assert "not a decided decision" in expect
    say("adopt: a failed anchor test, a missing run folder or a failed commit leave base config and champions/; anchor test passed, "
        "champions/toy/ written, the arm's settings in the base config, one commit; an undecided decision is refused")
    # E2: an untrained control -> INVALID
    e2 = json.loads(research(["new", "toy", "broken_control", "a control that did not train"], env).stdout)["folder"]
    write_card(root, e2, {"id": "E2", "title": "broken control", "question": "q", "hypothesis": "h", "prediction": 0.1, "control": "c", "seeds": [0],
                          "arms": {"c": {"set": {**quick, "steps": 1, "lr": 1e-9, "warmup": 1, "save_every": 1, "eval_every": 1}}, "a": {"set": quick}}, "rule": {"boot": 200}})
    wait(root, e2, json.loads(research(["start", "toy", "E2"], env).stdout)["started"])
    v2 = json.loads(research(["judge", "toy", "E2", "--write", "--boot", "200"], env).stdout)
    assert v2["arms"]["a"]["verdict"] == "INVALID" and "floor" in v2["arms"]["a"]["reason"], v2
    assert v2["judge_code"]["files"] and "autoresearch/research/compute.py" in v2["judge_code"]["from_snapshot"], v2["judge_code"]
    say("an untrained control makes the verdict INVALID, not KEEP")
    # E3: a kill rule fires; E5 (next to it): a kill rule on a later row of a step that writes into its own folder (problem.json "dir"), the
    # stop through CONTROL there (exit 75); E4: stop and resume
    e3, e4, e5 = (json.loads(research(["new", "toy", slug, title], env).stdout)["folder"]
                  for slug, title in (("kill", "kill rule"), ("long", "stop and resume"), ("stage", "a step with its own folder")))
    write_card(root, e3, {"id": "E3", "title": "kill", "question": "q", "hypothesis": "h", "kind": "measure", "seeds": [0],
                          "arms": {"k": {"set": {**quick, "steps": 3000, "log_every": 1, "save_every": 500}}}, "kill": [{"metric": "total", "after_step": 5, "above": 0.0}]})
    write_card(root, e5, {"id": "E5", "title": "stage", "question": "q", "hypothesis": "h", "kind": "measure", "seeds": [0], "arms": {"s": {}},
                          "steps": ["stage"], "kill": [{"metric": "total", "after_step": 5, "above": 0.5}]})
    research(["start", "toy", "E5"], env)
    st3 = wait(root, e3, json.loads(research(["start", "toy", "E3"], env).stdout)["started"])
    assert st3 == {"E3.k_s0": "killed"}, st3
    assert wait(root, e5, ["E5.s_s0"], seconds=120) == {"E5.s_s0": "killed"}, run_trace(tmp, root, e5, "E5.s_s0")
    s5 = json.load(open(os.path.join(root, e5, "runs", "E5.s_s0", "run.json")))["steps"]["stage"]
    assert s5["exit"] == 75 and "at step 20" in s5["why"], s5
    say("a kill rule stops the training and marks the run killed; also on a later row, in a step's own folder (log.jsonl and CONTROL there)")
    write_card(root, e4, {"id": "E4", "title": "long", "question": "q", "hypothesis": "h", "kind": "measure", "seeds": [0],
                          "arms": {"m": {"set": {**quick, "steps": 4000, "log_every": 1, "save_every": 500, "eval_every": 4000}}}})
    research(["start", "toy", "E4"], env)
    logf = os.path.join(tmp, "runs", "toy", "E4.m_s0", "log.jsonl")
    end = time.time() + 300
    while not (os.path.exists(logf) and os.path.getsize(logf) > 0) and time.time() < end:
        time.sleep(0.2)
    assert json.loads(research(["stop", "toy", "E4"], env).stdout)["stop requested"] == ["E4.m_s0"]
    assert wait(root, e4, ["E4.m_s0"]) == {"E4.m_s0": "stopped"}
    assert json.loads(research(["resume", "toy", "E4"], env).stdout)["resumed"] == ["E4.m_s0"]
    assert wait(root, e4, ["E4.m_s0"]) == {"E4.m_s0": "ok"}
    m4 = json.loads(research(["judge", "toy", "E4"], env).stdout)
    assert m4["arms"]["m"]["verdict"] == "MEASURED" and m4["arms"]["m"]["seeds"]["0"]["numbers"]["seg.iou"] > 0.3, m4
    assert "no DONE verdict" in research(["register", "toy", "E4", "m", "--decision", "D1", "--role", "comparison"], env, ok=(1,)).stderr
    say("stop (exit 75, status stopped) and resume (the finished steps skipped, the training continued) to the end; a measurement")
    # holdout, an evaluation by hand, TEST without a log row
    hand = sh([sys.executable, "-m", "research.evaluate", "toy", "E1.a_s0"], os.path.join(root, "problems", "toy"), dict(env, PYTHONPATH=os.pathsep.join([os.path.join(root, "problems", "toy"), core])), ok=(1,))
    assert "inside a worker" in hand.stderr
    assert "TEST refused" in research(["test", "toy", "E1.a_s0"], env, ok=(1,)).stderr
    code = os.path.join(tmp, "runs", "toy", "_code", snap[:12])
    probe = ("import os,sys; from Evaluation import Evaluation\n"
             "try:\n    Evaluation(project=os.getcwd()).run(sys.argv[1], 'test', metrics=('seg',), device='cpu')\n"
             "except Exception as e:\n    print('REFUSED', e); sys.exit(0)\nsys.exit(1)\n")
    penv = dict(env, PYTHONPATH=os.pathsep.join([os.path.join(code, "problems", "toy"), os.path.join(code, "autoresearch")]), RESEARCH_RUNS=os.path.join(tmp, "runs", "toy"))
    read_test = lambda e: sh([sys.executable, "-c", probe, os.path.join(tmp, "runs", "toy", "E1.a_s0")], os.path.join(code, "problems", "toy"), e).stdout  # noqa: E731
    assert "is a holdout split" in read_test(penv)
    as_test = dict(penv, RESEARCH_REPO=root, RESEARCH_PROBLEM="toy", RESEARCH_RUN="E1.a_s0", RESEARCH_HOLDOUT_RUN="E1.a_s0")      # set by hand
    assert "is a holdout split" in read_test(as_test), "the variables of a TEST run without a row of the TEST log"
    test_log = os.path.join(root, "problems", "toy", "TEST_LOG.md")
    keep_log = open(test_log, encoding="utf-8").read()
    open(test_log, "a", encoding="utf-8").write("| 2026-11-27 | E1.a_s0 | D1 | selftest |\n")
    opened = read_test(as_test)
    open(test_log, "w", encoding="utf-8").write(keep_log)
    assert "REFUSED" in opened and "is a holdout split" not in opened, opened         # the gate opens; the toy has no test shard to read
    say("refused: an evaluation step by hand, TEST without a log row, a holdout read outside a TEST run or without a row of the TEST log")
    # the hook: a protected file without and with a grant
    hook = os.path.join(core, "research", "hook.py")
    ev = {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(root, "problems", "toy", "problem.json")}, "cwd": root}
    r = subprocess.run([sys.executable, hook], input=json.dumps(ev), capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "protected" in r.stderr, (r.returncode, r.stderr)
    ok = dict(ev, tool_input={"file_path": os.path.join(root, "problems", "toy", "net", "model", "net.py")})
    assert subprocess.run([sys.executable, hook], input=json.dumps(ok), capture_output=True, text=True, env=env).returncode == 0
    mcp = {"tool_name": "mcp__files__write_file", "tool_input": {"path": os.path.join(root, "problems", "toy", "TEST_LOG.md")}, "cwd": root}
    assert subprocess.run([sys.executable, hook], input=json.dumps(mcp), capture_output=True, text=True, env=env).returncode == 2, "an MCP file tool is guarded too"
    bare = os.path.join(root, "problems", "bare")                   # a problem whose problem.json lists nothing as protected
    os.makedirs(bare)
    toy_p = json.load(open(os.path.join(root, "problems", "toy", "problem.json"), encoding="utf-8"))
    json.dump({k: v for k, v in toy_p.items() if k != "protected"} | {"name": "bare"}, open(os.path.join(bare, "problem.json"), "w"))
    grants_json = os.path.join(os.path.dirname(machine), "state", "grants.json")
    os.makedirs(os.path.dirname(grants_json), exist_ok=True)
    json.dump([1, "x", {"path": 3}], open(grants_json, "w"))         # a grants file with entries that are no grants: it grants nothing
    hook_run = lambda path: subprocess.run([sys.executable, hook], input=json.dumps(dict(ev, tool_input={"file_path": path})), capture_output=True,  # noqa: E731
                                           text=True, env=env)
    for path, why in ((grants_json, "state folder"), (os.path.join(root, "problems", "toy", "test_log.md"), "is protected"),
                      (os.path.join(bare, "TEST_LOG.md"), "is protected"), (os.path.join(root, "problems", "toy", "TEST_LOG.md::$DATA"), "is protected"),
                      (os.path.join(root, "champions", "toy", "champions.json"), "is protected")):
        r = hook_run(path)
        assert r.returncode == 2 and why in r.stderr, (path, r.returncode, r.stderr)
    for module, cases in (("gates.py", ((test_log, 2), (net_py, 0))), ("paths.py", ((os.path.join(tmp, "outside.txt"), 0), (test_log, 2)))):
        mod = os.path.join(core, "research", module)                # a broken gate module: the static list decides, the rest goes through
        keep_mod = open(mod, encoding="utf-8").read()
        open(mod, "a", encoding="utf-8").write("\n)(\n")
        got = [(path, want, hook_run(path)) for path, want in cases]
        open(mod, "w", encoding="utf-8").write(keep_mod)
        for path, want, r in got:
            assert r.returncode == want and (want == 0 or "git checkout" in r.stderr), (module, path, r.returncode, r.stderr)
    research(["grant", "problems/toy/problem.json", "--decision", "D1", "--hours", "1", "--note", "selftest"], env)
    assert subprocess.run([sys.executable, hook], input=json.dumps(ev), capture_output=True, text=True, env=env).returncode == 0
    assert "not a decided" in research(["grant", "problems/toy/TEST_LOG.md", "--decision", "D9", "--note", "x"], env, ok=(1,)).stderr
    say("hook: a protected file is refused (any case, a stream name, a TEST log its problem does not list, the state folder, a grants file of "
        "no grants), allowed under a grant of a decided decision; an undecided grant is refused; with a broken gates.py or paths.py a static "
        "list decides (TEST_LOG.md refused, a source file and a file outside allowed)")
    open(os.path.join(root, e1, "runs", "README.md"), "w").write("not a run\n")
    st = research(["status", "toy"], env).stdout
    assert "E1.a_s0" in st and "E4.m_s0" in st
    tr = json.loads(research(["tests", "toy"], env).stdout)
    assert tr["passed"] and [t["test"] for t in tr["tests"]] == ["tests/test_toy.py --quick"], tr
    say("tests: the problem's own tests with its interpreter and module path")
    # clean: E2 is judged (INVALID): its runs go on the list; E3 has a KEEP arm, E5 a MEASURED one, E4's step lives, E1.a is the champion:
    # they stay; E1's control goes too when E1's arm is CUT (KEEP, INCONCLUSIVE and TIE keep it)
    runs = os.path.join(tmp, "runs", "toy")
    for folder, arms in ((e3, {"k": {"verdict": "KEEP"}}), (e4, {"m": {"verdict": "CUT"}}), (e5, {"s": {"verdict": "MEASURED"}})):
        json.dump({"verdict": "DONE", "arms": arms}, open(os.path.join(root, folder, "verdict.json"), "w"))
    reg = research(["register", "toy", "E5", "s", "--decision", "D1", "--role", "comparison"], env, ok=(1,)).stderr     # E5 has no result.json
    assert "refused: E5.s_s0: its records cannot be read" in reg, reg
    alive = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        open(os.path.join(runs, "E4.m_s0", "step.pid"), "w").write(str(alive.pid))
        listed = json.loads(research(["clean", "toy"], env).stdout)
        expect = {"E2.a_s0", "E2.c_s0"} | ({"E1.c_s0", "E1.c_s1"} if arm["verdict"] == "CUT" else set())
        assert {i["run"] for i in listed["runs"]} == expect and os.path.isfile(listed["list"]), listed
        kept = listed["kept"]
        assert "step still runs" in kept["E4.m_s0"] and "KEEP" in kept["E3.k_s0"] and "MEASURED" in kept["E5.s_s0"] and "champion" in kept["E1.a_s0"], kept
    finally:
        alive.kill()
        alive.wait()
    # after the listing: a new card builds on E2.c (it must stay), E4's step ended (deletable now, but not on the list)
    e6 = json.loads(research(["new", "toy", "reuse", "builds on E2.c"], env).stdout)["folder"]
    write_card(root, e6, {"id": "E6", "title": "reuse", "question": "q", "hypothesis": "h", "kind": "measure", "seeds": [0], "arms": {"r": {"from": "E2.c"}}})
    h = listed["hash"][:12]
    unbound = research(["clean", "toy", "--delete", listed["list"], "--decision", "D1"], env, ok=(1,)).stderr
    assert "does not name this list" in unbound and h in unbound, unbound       # a decided decision that does not name the list's hash
    from research.upkeep import _list_hash                          # a list written by hand, with a non-run id after a deletable run
    crafted = {"problem": "toy", "made": "by hand", "runs": [{"run": "E2.a_s0", "bytes": 1}, {"run": "../x", "bytes": 0}], "bytes": 1}
    crafted["hash"] = _list_hash(crafted)
    crafted_path = os.path.join(tmp, "crafted.json")
    json.dump(crafted, open(crafted_path, "w"))
    decisions = os.path.join(root, "DECISIONS.md")
    owner = "| 2026-11-27 | D1: the selftest's owner says yes |\n"
    text = open(decisions, encoding="utf-8").read()
    assert owner in text
    with open(decisions, "w", encoding="utf-8") as f:
        f.write(text.replace(owner, owner + f"| 2026-11-27 | D2: delete the runs of clean list {h} |\n"
                             f"| 2026-11-27 | D3: delete the runs of clean list {crafted['hash'][:12]} |\n"))
    assert "no runs" in research(["clean", "toy", "--delete", crafted_path, "--decision", "D3"], env, ok=(1,)).stderr
    assert os.path.isdir(os.path.join(runs, "E2.a_s0")), "a list is checked as a whole before anything is deleted"
    champs = os.path.join(root, "champions", "toy", "champions.json")
    keep_champs = open(champs, encoding="utf-8").read()
    open(champs, "w").write("{")
    assert "cannot be read" in research(["clean", "toy", "--delete", listed["list"], "--decision", "D2"], env, ok=(1,)).stderr
    open(champs, "w", encoding="utf-8").write(keep_champs)
    done = json.loads(research(["clean", "toy", "--delete", listed["list"], "--decision", "D2"], env).stdout)
    assert set(done["deleted"]) == expect - {"E2.c_s0"} and list(done["kept"]) == ["E2.c_s0"] and "E006" in done["kept"]["E2.c_s0"], done
    assert not os.path.exists(os.path.join(runs, "E2.a_s0")) and os.path.isdir(os.path.join(runs, "E2.c_s0")) and os.path.isdir(os.path.join(runs, "E4.m_s0"))
    say("clean: a list (a live step, KEEP and MEASURED experiments, the champion kept), then exactly that list under the decided row that names "
        "its hash, checked as a whole first and each run again; another decision, a list by hand and an unreadable champions.json refuse")


def main(argv) -> int:
    sys.path.insert(0, CORE)
    tmp = tempfile.mkdtemp(prefix="research_selftest_")
    try:
        parts(tmp)
        if "--quick" not in argv:
            try:                                             # the chain trains the toy with this interpreter (the machine file names it)
                import numpy, torch  # noqa: F401,E401
            except ImportError as e:
                raise SystemExit(f"the chain needs numpy and torch ({e}): run it with the problems' python, e.g. ~/miniconda3/envs/gcar/bin/python -m research.selftest")
            chain(tmp)
    finally:
        if "--keep" not in argv:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"kept {tmp}")
    print("research selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

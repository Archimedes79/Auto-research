"""The test of the research system: known answers of its parts, then the whole chain on the toy problem in a throw-away repository (nothing of
this checkout and no machine state is touched).

    python -m research.selftest [--quick]        (from autoresearch/, with the interpreter that has torch; --quick: the parts only, seconds)

The chain: new -> check -> start (snapshot, tag, workers) -> runs from the snapshot -> judge (verdict, ledger, commit) -> adopt (anchor) and the
refusals: a card changed after its start, an untrained control (INVALID), a kill rule, stop and resume, a holdout read, an evaluation by hand,
an edit of a protected file without and with a grant."""
from __future__ import annotations

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
    from research import gates, judge, spec, worker
    from research.paths import parse_run, run_id
    p = json.load(open(os.path.join(REPO, "problems", "toy", "problem.json"), encoding="utf-8"))
    assert spec.problem_errors(p, "toy") == []
    bad = dict(p, scroe=1)
    assert any("unknown key 'scroe'" in e for e in spec.problem_errors(bad, "toy"))
    base = json.load(open(os.path.join(REPO, "problems", "toy", "training", "configs", "base.json"), encoding="utf-8"))
    card = {"id": "E1", "title": "t", "question": "q", "hypothesis": "h", "prediction": 0.02, "control": "c", "seeds": [0, 1],
            "arms": {"c": {"set": {"width": 4}}, "a": {"set": {"widht": 8}}}}
    errs = spec.card_errors(card, p, base)
    assert any("'widht' is not a setting" in e for e in errs) and len(errs) == 1, errs
    assert any("holdout" in e for e in spec.card_errors(dict(card, split="test", arms={"c": {}, "a": {}}), p, base))
    assert any("control" in e for e in spec.card_errors(dict(card, control="x", arms={"c": {}, "a": {}}), p, base))
    measure = {**{k: v for k, v in card.items() if k not in ("control", "prediction")}, "kind": "measure", "arms": {"m": {}}}
    assert spec.card_errors(measure, p, base) == []
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
    # kill rules on a growing log
    log = os.path.join(tmp, "log.jsonl")
    k = worker.Kill([{"metric": "total", "after_step": 20, "above": 0.5}], log)
    with open(log, "w") as f:
        f.write(json.dumps({"step": 10, "total": 0.9}) + "\n" + '{"step": 20, "tot')
    assert k.fired() == ""
    with open(log, "a") as f:
        f.write('al": 0.7}\n')
    assert "kill rule" in k.fired()
    # decisions and the TEST log
    root = os.path.join(tmp, "repo_g")
    os.makedirs(os.path.join(root, "problems", "x"))
    json.dump(p | {"name": "x"}, open(os.path.join(root, "problems", "x", "problem.json"), "w"))
    open(os.path.join(root, "DECISIONS.md"), "w").write("# D\n## Decided\n| date | decision |\n|---|---|\n| 2026-11-27 | D7: ok |\n\n## Open\n| D8 | asked |\n")
    assert gates.decided("x", root) == {"D7"}
    log_path = os.path.join(root, "problems", "x", "TEST_LOG.md")
    open(log_path, "w").write("# TEST\n## Runs\n| date | run | decision | note |\n|---|---|---|---|\n| 2026-11-27 | E1.a_s0 | D8 | asked |\n| 2026-11-27 | E1.c_s0 | D7 | once |\n")
    assert "not a decided" in gates.test_refusal("x", "E1.a_s0", root)
    assert gates.test_refusal("x", "E1.c_s0", root) == ""
    assert "not named" in gates.test_refusal("x", "E1.b_s0", root)
    assert gates.is_protected("problems/x/problem.json", root) and gates.is_protected("problems/x/evaluation/adapter.py", root)
    assert gates.is_protected("champions/x/E1.a/network.json", root) and not gates.is_protected("problems/x/net/model/net.py", root)
    # a card may only gain seeds after its start; nested settings merge key by key; results scored differently are not comparable
    assert spec.same_rule(card, dict(card, seeds=[0, 1, 2])) and not spec.same_rule(card, dict(card, seeds=[1, 0])) and not spec.same_rule(card, dict(card, prediction=0.1))
    assert spec.merged({"m": {"a": 1, "b": 2}, "c": 3}, {"m": {"b": 5}}) == {"m": {"a": 1, "b": 5}, "c": 3}
    assert spec.unknown_settings({"m": {"x": 1}}, {"m": {"a": 1}}) == ["m.x"]
    r = {"split": "val", "frames": 160, "evaluation": {"gt": "g1", "evaluator": "e1", "metrics": ["seg"]}}
    assert judge.incomparable([{"result": r}, {"result": r}]) == []
    assert any("ground truth" in x for x in judge.incomparable([{"result": r}, {"result": dict(r, evaluation=dict(r["evaluation"], gt="g2"))}]))
    say("parts: schemas and typos, compose, run ids, the standard rule, kill rules, decisions, the TEST log, protected paths")


# ------------------------------------------------------------------------------------------------ the chain
def sh(cmd, cwd, env, ok=(0,)):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode in ok, f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r


def research(args, env, ok=(0,)):
    return sh([sys.executable, "-m", "research", "--json", *args], env["RESEARCH_TMP_CORE"], env, ok)


def make_repo(tmp: str) -> str:
    root = os.path.join(tmp, "repo")
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "temp", "eval_out", "state", "history")
    for d in ("research", "Evaluation", "trainkit", "templates"):
        shutil.copytree(os.path.join(CORE, d), os.path.join(root, "autoresearch", d), ignore=skip)
    shutil.copytree(os.path.join(REPO, "problems", "toy"), os.path.join(root, "problems", "toy"), ignore=shutil.ignore_patterns("__pycache__", "experiments", "ledger.csv"))
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
    card = {"id": "E1", "title": "width", "question": "does a wider network find the disc better?", "hypothesis": "more channels, better masks",
            "prediction": 0.05, "control": "c", "seeds": [0, 1],
            "arms": {"c": {"set": {**quick, "steps": 300, "width": 8}}, "a": {"set": {**quick, "steps": 300, "width": 12}}},
            "rule": {"boot": 300}, "kill": [{"metric": "total", "after_step": 10, "above": 100.0}]}
    write_card(root, e1, card)
    assert json.loads(research(["check", "toy", "E1"], env).stdout)["errors"] == []
    net_py = os.path.join(root, "problems", "toy", "net", "model", "net.py")
    keep = open(net_py, encoding="utf-8").read()
    open(net_py, "a", encoding="utf-8").write("# a half-done edit of another chat\n")
    assert any("not committed" in e for e in json.loads(research(["check", "toy", "E1"], env).stdout)["errors"])
    assert "not committed" in research(["start", "toy", "E1"], env, ok=(1,)).stderr
    open(net_py, "w", encoding="utf-8").write(keep)
    started = json.loads(research(["start", "toy", "E1"], env).stdout)
    assert sorted(started["started"]) == ["E1.a_s0", "E1.a_s1", "E1.c_s0", "E1.c_s1"] and started["tag"] == "toy/E1", started
    snap = subprocess.run(["git", "rev-parse", "toy/E1^{commit}"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert snap and subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() != snap
    st = wait(root, e1, started["started"])
    assert set(st.values()) == {"ok"}, (st, open(os.path.join(tmp, "runs", "toy", "E1.c_s0", "train.log")).read()[-1500:])
    res = json.load(open(os.path.join(root, e1, "runs", "E1.a_s0", "result.json")))
    assert res["identity"]["commit"] == snap and res["split"] == "val" and res["frames"] == 160 and "seg.iou" in res["numbers"], res
    say(f"start: one snapshot {started['tag']}, four runs from it, results (E1.a_s0 IoU {res['score']['value']:.3f})")
    # a second start of the same experiment starts nothing and keeps the tag
    again = json.loads(research(["start", "toy", "E1"], env).stdout)
    assert again["started"] == [] and len(again["skipped"]) == 4 and again["commit"] == snap[:12]
    # the card may not change after its start
    text = open(os.path.join(root, e1, "experiment.md")).read()
    write_card(root, e1, dict(card, rule={"boot": 300, "min_effect": 0.5}))
    assert any("changed since its snapshot" in e for e in json.loads(research(["check", "toy", "E1"], env).stdout)["errors"])
    assert "changed since its snapshot" in research(["judge", "toy", "E1"], env, ok=(1,)).stderr
    open(os.path.join(root, e1, "experiment.md"), "w").write(text + "\nThe conclusion may be written; the card block may not change.\n")
    v = json.loads(research(["judge", "toy", "E1", "--write", "--boot", "300"], env).stdout)
    arm = v["arms"]["a"]
    assert arm["verdict"] in ("KEEP", "TIE", "CUT", "INCONCLUSIVE") and len(arm["interval"]) == 2 and v["records_commit"] and v["snapshot"] == snap, v
    log = subprocess.run(["git", "log", "-1", "--name-only", "--format=%s"], cwd=root, capture_output=True, text=True).stdout
    assert "verdict" in log and "problems/toy/ledger.csv" in log and "verdict.json" in log, log
    assert json.loads(research(["index", "toy", "--check"], env).stdout)["current"]
    say(f"judge: {arm['verdict']} (gain {arm['gain']:+.3f} [{arm['interval'][0]:+.3f}, {arm['interval'][1]:+.3f}]), verdict.json, ledger, one commit")
    # adopt (by hand when it was no KEEP): the anchor evaluates again with the working tree's code and must agree bit for bit
    by_hand = [] if arm["verdict"] == "KEEP" else ["--by-hand", "the selftest adopts whatever won"]
    ad = json.loads(research(["adopt", "toy", "E1", "a", "--decision", "D1", *by_hand], env).stdout)
    champ = json.load(open(os.path.join(root, "champions", "toy", "champions.json")))
    assert ad["champion"] == "E1.a" and champ["champion"] == "E1.a" and ad["commit"]
    assert json.load(open(os.path.join(root, "problems", "toy", "training", "configs", "base.json")))["width"] == 12
    expect = research(["adopt", "toy", "E1", "a", "--decision", "D2"], env, ok=(1,)).stderr
    assert "not a decided decision" in expect
    say("adopt: anchor test passed, champions/toy/ written, the arm's settings in the base config, one commit; an undecided decision is refused")
    # E2: an untrained control -> INVALID
    e2 = json.loads(research(["new", "toy", "broken_control", "a control that did not train"], env).stdout)["folder"]
    write_card(root, e2, {"id": "E2", "title": "broken control", "question": "q", "hypothesis": "h", "prediction": 0.1, "control": "c", "seeds": [0],
                          "arms": {"c": {"set": {**quick, "steps": 1, "lr": 1e-9, "warmup": 1, "save_every": 1, "eval_every": 1}}, "a": {"set": quick}}, "rule": {"boot": 200}})
    wait(root, e2, json.loads(research(["start", "toy", "E2"], env).stdout)["started"])
    v2 = json.loads(research(["judge", "toy", "E2", "--boot", "200"], env).stdout)
    assert v2["arms"]["a"]["verdict"] == "INVALID" and "floor" in v2["arms"]["a"]["reason"], v2
    say("an untrained control makes the verdict INVALID, not KEEP")
    # E3: a kill rule fires; E4: stop and resume
    e3 = json.loads(research(["new", "toy", "kill", "kill rule"], env).stdout)["folder"]
    write_card(root, e3, {"id": "E3", "title": "kill", "question": "q", "hypothesis": "h", "kind": "measure", "seeds": [0],
                          "arms": {"k": {"set": {**quick, "steps": 3000, "log_every": 1, "save_every": 500}}}, "kill": [{"metric": "total", "after_step": 5, "above": 0.0}]})
    st3 = wait(root, e3, json.loads(research(["start", "toy", "E3"], env).stdout)["started"])
    assert st3 == {"E3.k_s0": "killed"}, st3
    say("a kill rule stops the training and marks the run killed")
    e4 = json.loads(research(["new", "toy", "long", "stop and resume"], env).stdout)["folder"]
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
    say("stop (exit 75, status stopped) and resume (the finished steps skipped, the training continued) to the end; a measurement")
    # holdout, an evaluation by hand, TEST without a log row
    hand = sh([sys.executable, "-m", "research.evaluate", "toy", "E1.a_s0"], os.path.join(root, "problems", "toy"), dict(env, PYTHONPATH=os.pathsep.join([os.path.join(root, "problems", "toy"), core])), ok=(1,))
    assert "inside a worker" in hand.stderr
    assert "TEST refused" in research(["test", "toy", "E1.a_s0"], env, ok=(1,)).stderr
    code = os.path.join(tmp, "runs", "toy", "_code", snap[:12])
    probe = ("import os,sys; from Evaluation import Evaluation\n"
             "try:\n    Evaluation(project=os.getcwd()).run(sys.argv[1], 'test', metrics=('seg',), device='cpu')\n"
             "except Exception as e:\n    print('REFUSED', e); sys.exit(0)\nsys.exit(1)\n")
    r = sh([sys.executable, "-c", probe, os.path.join(tmp, "runs", "toy", "E1.a_s0")], os.path.join(code, "problems", "toy"),
           dict(env, PYTHONPATH=os.pathsep.join([os.path.join(code, "problems", "toy"), os.path.join(code, "autoresearch")]), RESEARCH_RUNS=os.path.join(tmp, "runs", "toy")))
    assert "holdout" in r.stdout, r.stdout
    say("refused: an evaluation step by hand, TEST without a log row, a holdout read outside a TEST run")
    # the hook: a protected file without and with a grant
    hook = os.path.join(core, "research", "hook.py")
    ev = {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(root, "problems", "toy", "problem.json")}, "cwd": root}
    r = subprocess.run([sys.executable, hook], input=json.dumps(ev), capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "protected" in r.stderr, (r.returncode, r.stderr)
    ok = dict(ev, tool_input={"file_path": os.path.join(root, "problems", "toy", "net", "model", "net.py")})
    assert subprocess.run([sys.executable, hook], input=json.dumps(ok), capture_output=True, text=True, env=env).returncode == 0
    mcp = {"tool_name": "mcp__files__write_file", "tool_input": {"path": os.path.join(root, "problems", "toy", "TEST_LOG.md")}, "cwd": root}
    assert subprocess.run([sys.executable, hook], input=json.dumps(mcp), capture_output=True, text=True, env=env).returncode == 2, "an MCP file tool is guarded too"
    research(["grant", "problems/toy/problem.json", "--decision", "D1", "--hours", "1", "--note", "selftest"], env)
    assert subprocess.run([sys.executable, hook], input=json.dumps(ev), capture_output=True, text=True, env=env).returncode == 0
    assert "not a decided" in research(["grant", "problems/toy/TEST_LOG.md", "--decision", "D9", "--note", "x"], env, ok=(1,)).stderr
    say("hook: a protected file is refused, allowed under a grant of a decided decision; an undecided grant is refused")
    st = research(["status", "toy"], env).stdout
    assert "E1.a_s0" in st and "E4.m_s0" in st


def main(argv) -> int:
    sys.path.insert(0, CORE)
    tmp = tempfile.mkdtemp(prefix="research_selftest_")
    try:
        parts(tmp)
        if "--quick" not in argv:
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

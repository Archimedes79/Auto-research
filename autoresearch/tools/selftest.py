"""Self-test of the autoresearch tools in a throw-away project with its own git repository (the real ledger is never touched).

    python autoresearch/tools/selftest.py        (needs git and bash, e.g. in the WSL)

Tests: run.py (done flag only on success, retry only on a CUDA unknown error, timeout, card check, TEST guard, --after waits / blocks / sees a
stalled run, snapshot commit + tag per run through a temporary index, a finished id is refused without --again, a failed or stalled id is resumed
and keeps its first tag, a planned ledger row keeps its prediction, GPU slots reclaim dead owners and block on live ones, no git identity refused),
ledger.py (concurrent notes under the lock), decide.py (typed scores and experiment files with cards: seeds, interval, guards, subsets, deterministic), experiment.py (check), check_numbers.py, schedule.py (next, reconcile), clean_temp.py.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

T = tempfile.mkdtemp(prefix="ar_selftest_")
os.makedirs(f"{T}/code")
open(f"{T}/code/a.py", "w").write("x = 1\n")
json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
               code_dirs=["code"], scratch="temp", live_code=["code"]), open(f"{T}/project.json", "w"))
open(f"{T}/TEST.md", "w").write("| S.test.ok | logged |\n")
env = dict(os.environ, AR_PROJECT=T)
G = ["git", "-c", "safe.directory=*", "-C", T]
subprocess.run(G[:3] + ["init", "-q", T], check=True)
for k, v in (("user.name", "selftest"), ("user.email", "selftest@example.invalid"), ("core.autocrlf", "false")):
    subprocess.run(G + ["config", k, v], check=True)


def go(rid, cmd, *extra):
    return subprocess.run([sys.executable, f"{HERE}/run.py", "--id", rid, "--hyp", "selftest", "--retries", "1", *extra, "--", *cmd], capture_output=True, text=True, env=env)


def tags():
    return subprocess.run(G + ["tag"], capture_output=True, text=True).stdout.split()


def rows():
    return {r["id"]: r for r in csv.DictReader(open(f"{T}/LEDGER.csv", encoding="utf-8"))}


try:
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", "S.ok", "--pred", "x >= 1", "--rule", "keep if x >= 1"], check=True, env=env, capture_output=True)
    ok = go("S.ok", ["bash", "-c", "echo hello"])
    bad = go("S.bad", ["bash", "-c", "echo boom; exit 3"])
    cuda = go("S.cuda", ["bash", "-c", "echo 'CUDA error: unknown error'; exit 1"])
    assert ok.returncode == 0 and os.path.exists(f"{T}/runs/S.ok/done"), ok.stdout + ok.stderr
    assert bad.returncode == 3 and not os.path.exists(f"{T}/runs/S.bad/done")
    assert cuda.returncode == 1 and "2 attempt" in cuda.stdout and not os.path.exists(f"{T}/runs/S.cuda/done")
    r = rows()
    assert r["S.ok"]["status"] == "ok" and r["S.bad"]["status"] == "failed"
    assert r["S.ok"]["prediction"] == "x >= 1", "a planned row must keep its prediction"
    # code: commit + tag per run, a resumed id keeps its tag, a code change gives a new commit
    assert "t/S.ok" in tags() and r["S.ok"]["commit"] not in ("", "no-git", "git-error"), (tags(), r["S.ok"]["commit"])
    first = r["S.ok"]["commit"]
    open(f"{T}/code/a.py", "w").write("x = 2\n")
    ok2 = go("S.ok2", ["bash", "-c", "echo again"])
    assert ok2.returncode == 0 and rows()["S.ok2"]["commit"] != first and "t/S.ok2" in tags()
    refused = go("S.ok", ["bash", "-c", "echo resume"])
    assert refused.returncode == 2 and "finished with exit 0" in refused.stdout, "a finished id must be refused without --again"
    again = go("S.ok", ["bash", "-c", "echo resume"], "--again")
    assert again.returncode == 0 and "tag existed" in rows()["S.ok"]["notes"] and "code changed since attempt 1" in rows()["S.ok"]["notes"], rows()["S.ok"]["notes"]
    # --after: waits for done flags, blocks when a dependency failed
    aft = go("S.after", ["bash", "-c", "echo after"], "--after", "S.ok,S.ok2")
    blk = go("S.blocked", ["bash", "-c", "echo never"], "--after", "S.bad")
    assert aft.returncode == 0 and blk.returncode == 3 and rows()["S.blocked"]["status"] == "blocked", (aft.stdout, blk.stdout)
    assert go("S.bad", ["bash", "-c", "echo retry; exit 0"]).returncode == 0 and "attempt after failed" in rows()["S.bad"]["notes"], "a failed id is resumed"
    # card check and TEST guard
    assert go("S.card", ["bash", "-c", "echo x"], "--require-card").returncode == 2
    assert go("S.test", ["bash", "-c", "echo x"], "--split", "test").returncode == 2
    assert go("S.test.ok", ["bash", "-c", "echo x"], "--split", "test").returncode == 0
    # retry only on a CUDA unknown error; timeout kills the command
    oom = go("S.oom", ["bash", "-c", "echo 'CUDA error: out of memory'; exit 1"])
    assert oom.returncode == 1 and "1 attempt" in oom.stdout, oom.stdout
    to = go("S.timeout", ["bash", "-c", "sleep 30"], "--timeout", "0.03")
    assert to.returncode == 124 and rows()["S.timeout"]["status"] == "timeout", (to.returncode, rows().get("S.timeout"))
    # a killed launcher leaves a stalled record: --after blocks on it, schedule.reconcile marks it failed, a new attempt is allowed
    import run as RUN
    stall = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.stall", "--hyp", "selftest", "--no-git", "--", "bash", "-c", "sleep 60"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if RUN.record(f"{T}/runs/S.stall").get("status") == "running":
            break
        import time
        time.sleep(0.1)
    pid = RUN.record(f"{T}/runs/S.stall").get("pid")
    stall.kill()
    stall.wait()
    if pid:
        try:
            os.kill(int(pid), 9)
        except OSError:
            pass
    stale = dict(env, AR_STALE_S="1")
    time.sleep(1.5)
    assert RUN.state_of(f"{T}/runs/S.stall") == "running", "fresh heartbeat without the env override"
    dep = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.dep", "--hyp", "selftest", "--no-git", "--after", "S.stall", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=stale)
    assert dep.returncode == 3 and "stalled" in dep.stdout, dep.stdout
    redo = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.stall", "--hyp", "selftest", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=stale)
    assert redo.returncode == 0 and "attempt after stalled" in rows()["S.stall"]["notes"], redo.stdout
    # GPU slots: dead owners are reclaimed, live owners block
    os.makedirs(f"{T}/runs/.gpu_slots/0")
    json.dump(dict(id="ghost", pid=999999, run_dir=f"{T}/runs/ghost"), open(f"{T}/runs/.gpu_slots/0/run.json", "w"))
    assert go("S.slot", ["bash", "-c", "echo gpu"], "--gpu", "--gpu-slots", "1").returncode == 0, "a slot of a dead owner is reclaimed"
    os.makedirs(f"{T}/runs/.gpu_slots/0")
    os.makedirs(f"{T}/runs/S.live")
    json.dump(dict(id="me", pid=os.getpid(), run_dir=f"{T}/runs/S.live"), open(f"{T}/runs/.gpu_slots/0/run.json", "w"))
    json.dump(dict(status="running"), open(f"{T}/runs/S.live/run.json", "w"))
    open(f"{T}/runs/S.live/heartbeat", "w").write("now")
    busy = go("S.busy", ["bash", "-c", "echo gpu"], "--gpu", "--gpu-slots", "1", "--wait-gpu", "0")
    assert busy.returncode == 1 and rows()["S.busy"]["status"] == "gpu_busy", busy.stdout
    shutil.rmtree(f"{T}/runs/.gpu_slots")
    # concurrent ledger writes lose nothing
    procs = [subprocess.Popen([sys.executable, f"{HERE}/ledger.py", "note", f"S.c{i}", "--result", f"r{i}"], env=env, stdout=subprocess.DEVNULL) for i in range(8)]
    assert all(p.wait() == 0 for p in procs)
    assert all(rows()[f"S.c{i}"]["result"] == f"r{i}" for i in range(8)), "a concurrent note was lost"
    # no git identity: refused before anything runs
    T2 = tempfile.mkdtemp(prefix="ar_selftest_noid_")
    os.makedirs(f"{T2}/code")
    json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", run_records="runs", code_dirs=["code"]), open(f"{T2}/project.json", "w"))
    subprocess.run(["git", "-c", "safe.directory=*", "-C", T2, "init", "-q"], check=True)
    noid = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.noid", "--hyp", "selftest", "--", "bash", "-c", "echo x"], capture_output=True, text=True,
                          env=dict(env, AR_PROJECT=T2, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="", GIT_COMMITTER_NAME=""))
    shutil.rmtree(T2, ignore_errors=True)
    assert noid.returncode == 2 and "git cannot commit" in noid.stdout and not os.path.exists(f"{T2}/runs/S.noid/run.json"), noid.stdout + noid.stderr
    # ledger note + render
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "note", "S.ok", "--result", "x 1", "--decision", "KEEP"], check=True, env=env, capture_output=True)
    assert "KEEP" in open(f"{T}/LEDGER.md", encoding="utf-8").read()
    # decide
    from decide import decide
    assert decide([0.746, 0.741], [0.725, 0.722])[0] == "KEEP"
    assert decide([0.730, 0.729], [0.725, 0.722])[0] == "TIE"
    assert decide([0.760, 0.715], [0.725, 0.722])[0] == "INCONCLUSIVE"
    assert decide([0.746], [0.725])[0] == "INCONCLUSIVE"
    assert decide([0.70, 0.69], [0.725, 0.722])[0] == "CUT"
    # decide from an experiment file and cards: KEEP, TIE, CUT (guard / rule), INCONCLUSIVE (one seed), subsets with at_least, deterministic + interval
    import decide as DC
    os.makedirs(f"{T}/experiments/E1_x", exist_ok=True)
    os.makedirs(f"{T}/runs/eval/v", exist_ok=True)
    os.makedirs(f"{T}/runs/eval/c", exist_ok=True)
    json.dump(dict(name="v", metrics=dict(Q=0.760, F=0.9), compare=dict(name="c", paired=dict(Q=dict(diff=0.03, lo=0.01, hi=0.05)))), open(f"{T}/runs/eval/v/val.json", "w"))
    json.dump(dict(name="c", metrics=dict(Q=0.730, F=0.9)), open(f"{T}/runs/eval/c/val.json", "w"))
    json.dump(dict(a=dict(Q=0.76, F=0.9), b=dict(Q=0.74, F=0.9), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9)), open(f"{T}/cards.json", "w"))
    base = dict(id="E1", title="t", kind="variant", metric="Q", direction="higher", band=0.01, prediction=dict(value=0.03, vs="c"), control="c",
                rule=[dict(metric="Q", min_effect=0.01), dict(metric="F", max_loss=0.01)], evaluate=dict(split="val", compare="c"))
    def verdict(**kw):
        exp = dict(base, **kw)
        json.dump(exp, open(f"{T}/experiments/E1_x/experiment.json", "w"))
        return DC.decide_experiment(exp, T)["verdict"]
    two = dict(a=dict(cards=["cards.json#a", "cards.json#b"]), c=dict(cards=["cards.json#ca", "cards.json#cb"]))
    assert verdict(arms=two) == "KEEP"
    assert verdict(arms=dict(a=dict(cards=["cards.json#a"]), c=dict(cards=["cards.json#ca"]))) == "INCONCLUSIVE", "one seed"
    assert verdict(arms=dict(a=dict(names=["v"]), c=dict(names=["c"])), deterministic=True) == "KEEP", "deterministic + interval excluding 0"
    assert verdict(arms=dict(a=dict(names=["v"]), c=dict(names=["c"]))) == "INCONCLUSIVE", "one seed without the deterministic flag"
    assert verdict(arms=two, rule=[dict(metric="Q", min_effect=0.01), dict(metric="F", max_loss=0.01), dict(metric="Q", min_effect=0.05)]) == "CUT", "rule not met"
    json.dump(dict(a=dict(Q=0.76, F=0.8), b=dict(Q=0.74, F=0.8), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9)), open(f"{T}/cards.json", "w"))
    assert verdict(arms=two) == "CUT", "guard violated"
    json.dump(dict(a=dict(Q=0.735, F=0.9), b=dict(Q=0.725, F=0.9), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9)), open(f"{T}/cards.json", "w"))
    assert verdict(arms=two, rule=[dict(metric="Q", min_effect=0.0)]) == "TIE"
    json.dump(dict(n=dict(a=dict(R=0.8), a1=dict(R=0.79), c=dict(R=0.7), c1=dict(R=0.71)), b=dict(a=dict(R=0.8), a1=dict(R=0.79), c=dict(R=0.7), c1=dict(R=0.71)),
                   x=dict(a=dict(R=0.5), a1=dict(R=0.5), c=dict(R=0.7), c1=dict(R=0.7))), open(f"{T}/sub.json", "w"))
    subs = dict(a=dict(cards={s: [f"sub.json#{s}/a", f"sub.json#{s}/a1"] for s in "nbx"}), c=dict(cards={s: [f"sub.json#{s}/c", f"sub.json#{s}/c1"] for s in "nbx"}))
    assert verdict(arms=subs, metric="R", rule=[dict(metric="R", min_effect=0.05, subsets=["n", "b", "x"], at_least=2)]) == "KEEP", "2 of 3 subsets, two seeds each"
    assert verdict(arms=subs, metric="R", rule=[dict(metric="R", min_effect=0.05, subsets=["n", "b", "x"])]) == "CUT", "all subsets needed"
    one = dict(a=dict(cards={s: f"sub.json#{s}/a" for s in "nbx"}), c=dict(cards={s: f"sub.json#{s}/c" for s in "nbx"}))
    assert verdict(arms=one, metric="R", rule=[dict(metric="R", min_effect=0.05, subsets=["n", "b", "x"], at_least=2)]) == "INCONCLUSIVE", "one seed per subset"
    import experiment as X
    exp = dict(base, arms=two)
    json.dump(exp, open(f"{T}/experiments/E1_x/experiment.json", "w"))
    assert X.check(exp, f"{T}/experiments/E1_x/experiment.json") == []
    assert any("control" in b for b in X.check(dict(exp, control="z"), f"{T}/experiments/E1_x/experiment.json"))
    assert any("no metric" in b for b in X.check(dict(exp, rule=[dict(metric="Q", min_effect=0.01), dict(metric="nope", max_loss=0.1)]), f"{T}/experiments/E1_x/experiment.json"))
    assert subprocess.run([sys.executable, f"{HERE}/decide.py", "--experiment", f"{T}/experiments/E1_x/experiment.json", "--write"], capture_output=True, text=True, env=env).returncode == 0
    assert os.path.exists(f"{T}/experiments/E1_x/decision.json")
    # check_numbers
    open(f"{T}/doc.md", "w").write("S 0.746 and 0.99 here")
    open(f"{T}/src.json", "w").write('{"S": 0.74612}')
    out = subprocess.run([sys.executable, f"{HERE}/check_numbers.py", f"{T}/doc.md", f"{T}/src.json"], capture_output=True, text=True).stdout
    assert "2 decimal numbers, 1 found" in out and "0.99" in out, out
    # schedule: next respects dependencies, reconcile reads the run records
    import schedule as SC
    sch = [dict(item="a", state="planned", depends_on="", priority="1", run_ids="S.ok2", resource="gpu"),
           dict(item="b", state="planned", depends_on="a", priority="1", run_ids="S.oom", resource="gpu"),
           dict(item="c", state="planned", depends_on="", priority="2", run_ids="", resource="cpu")]
    assert [x["item"] for x in SC.ready(sch)] == ["a", "c"]
    changed = SC.reconcile(sch, f"{T}/runs")
    assert ("a", "planned", "done") in changed and ("b", "planned", "failed") in changed, changed
    import clean_temp
    clean_temp.selftest()
    print("autoresearch tools selftest passed")
finally:
    shutil.rmtree(T, ignore_errors=True)

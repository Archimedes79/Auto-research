"""Self-test of the autoresearch tools in a throw-away project with its own git repository (the real ledger is never touched).

    python autoresearch/tools/selftest.py        (needs git and bash, e.g. in the WSL)

Tests: run.py (done flag only on success, retry only on a CUDA unknown error, timeout, card check, TEST guard, --after waits / blocks / sees a
stalled run, snapshot commit + tag per run through a temporary index, a finished id is refused without --again, a failed or stalled id is resumed
and keeps its first tag, a planned ledger row keeps its prediction, GPU slots reclaim dead owners and block on live ones, no git identity refused),
tick.py (dry run, derived states that are never written back, gpu slots from the slot directories, dependencies, commands from the schedule or the experiment arm, blocked items), ledger.py (concurrent notes under the lock, sync, tail with the runs in progress), decide.py (typed scores and experiment files with cards: seeds, interval, guards, subsets, deterministic), experiment.py (check), card.py (the evaluator contract, a benchmark script as a non-ML problem through the same loop, the holdout split from project.json), check_numbers.py, schedule.py (next, derived states), clean_temp.py.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

T = tempfile.mkdtemp(prefix="ar_selftest_")
os.makedirs(f"{T}/code")
open(f"{T}/code/a.py", "w").write("x = 1\n")
json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
               code_dirs=["code"], scratch="temp", live_code=["code"], cpu_slots=5), open(f"{T}/project.json", "w"))  # 5 here; the cpu-cap test below uses the default 1
open(f"{T}/TEST.md", "w").write("| S.test.ok | logged |\n")
env = dict(os.environ, AR_PROJECT=T)
G = ["git", "-c", "safe.directory=*", "-C", T]
subprocess.run(G[:3] + ["init", "-q", T], check=True)
for k, v in (("user.name", "selftest"), ("user.email", "selftest@example.invalid"), ("core.autocrlf", "false")):
    subprocess.run(G + ["config", k, v], check=True)


def go(rid, cmd, *extra):
    return subprocess.run([sys.executable, f"{HERE}/run.py", "--id", rid, "--hyp", "selftest", "--pred", "x >= 1", "--rule", "keep if x >= 1", "--retries", "1", *extra, "--", *cmd], capture_output=True, text=True, env=env)


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
    nocard = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.card", "--hyp", "h", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)
    assert nocard.returncode == 2 and "refused: a run needs its card" in nocard.stdout, nocard.stdout
    smoke = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.smoke", "--no-card", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)
    assert smoke.returncode == 0 and rows()["S.smoke"]["notes"] == "no card", (smoke.stdout, rows().get("S.smoke"))
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", "S.planned", "--hyp", "planned", "--pred", "y >= 2", "--rule", "keep if y >= 2"], check=True, env=env, capture_output=True)
    pl = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.planned", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)
    assert pl.returncode == 0 and rows()["S.planned"]["prediction"] == "y >= 2" and rows()["S.planned"]["hypothesis"] == "planned", pl.stdout
    assert go("S.test", ["bash", "-c", "echo x"], "--split", "test").returncode == 2
    assert go("S.test.ok", ["bash", "-c", "echo x"], "--split", "test").returncode == 0
    # retry only on a CUDA unknown error; timeout kills the command
    oom = go("S.oom", ["bash", "-c", "echo 'CUDA error: out of memory'; exit 1"])
    assert oom.returncode == 1 and "1 attempt" in oom.stdout, oom.stdout
    to = go("S.timeout", ["bash", "-c", "sleep 30"], "--timeout", "0.03")
    assert to.returncode == 124 and rows()["S.timeout"]["status"] == "timeout", (to.returncode, rows().get("S.timeout"))
    # a killed launcher leaves a stalled record: --after blocks on it, schedule.derived shows it failed, a new attempt is allowed
    import run as RUN
    stall = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.stall", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "sleep 60"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    dep = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.dep", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--after", "S.stall", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=stale)
    assert dep.returncode == 3 and "stalled" in dep.stdout, dep.stdout
    redo = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.stall", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=stale)
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
    # a slot directory without an owner file (taker died between mkdir and the write) is reclaimed after 10 s
    os.makedirs(f"{T}/runs/.gpu_slots/0")
    old_t = time.time() - 60
    os.utime(f"{T}/runs/.gpu_slots/0", (old_t, old_t))
    assert go("S.slot2", ["bash", "-c", "echo gpu"], "--gpu", "--gpu-slots", "1").returncode == 0, "an ownerless old slot is reclaimed"
    # --again clears the earlier attempt's result and decision
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "note", "S.ok2", "--decision", "KEEP"], check=True, env=env, capture_output=True)
    assert go("S.ok2", ["bash", "-c", "echo again"], "--again").returncode == 0 and rows()["S.ok2"]["decision"] == "", rows()["S.ok2"]["decision"]
    # the heartbeat is kept while a launcher waits for --after (a waiting chain member is not stalled)
    fast = dict(env, AR_STALE_S="1", AR_HEARTBEAT_S="0.3")
    pa = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.slowA", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "sleep 4"], env=fast, stdout=subprocess.DEVNULL)
    time.sleep(0.5)
    pb = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.waitB", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--after", "S.slowA", "--", "bash", "-c", "echo b"], env=fast, stdout=subprocess.DEVNULL)
    time.sleep(2.5)
    assert RUN.record(f"{T}/runs/S.waitB").get("status") == "waiting" and RUN.heartbeat_age(f"{T}/runs/S.waitB") < 1.5, "a waiting launcher must beat"
    assert pa.wait() == 0 and pb.wait() == 0
    # a ledger that stays busy does not kill the launcher (its exit code is the command's); `ledger.py sync` adds the row from the run record afterwards
    open(f"{T}/LEDGER.csv.lock", "w").write("held")
    held = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.locked", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=dict(env, AR_LOCK_WAIT_S="1"))
    os.remove(f"{T}/LEDGER.csv.lock")
    assert held.returncode == 0 and "ledger not written" in held.stdout and "S.locked" not in rows(), held.stdout
    dry_sync = subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync", "--dry-run"], capture_output=True, text=True, env=env)
    assert "would add S.locked" in dry_sync.stdout and "S.locked" not in rows(), dry_sync.stdout
    sy = subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync"], capture_output=True, text=True, env=env)
    assert "added S.locked" in sy.stdout and rows()["S.locked"]["status"] == "ok" and rows()["S.locked"]["exit"] == "0", sy.stdout + sy.stderr
    assert "0 row(s) added" in subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync"], capture_output=True, text=True, env=env).stdout, "sync is idempotent"
    # clean_temp tags the scratch through the launcher's snapshot (the call with the new signature)
    os.makedirs(f"{T}/temp/loop98")
    open(f"{T}/temp/loop98/scratch98.py", "w").write("y = 1")
    ct = subprocess.run([sys.executable, f"{HERE}/clean_temp.py", "loop98", "--move", "--force"], capture_output=True, text=True, env=env)
    assert ct.returncode == 0 and "t/loop98-scratch" in tags(), ct.stdout + ct.stderr
    # concurrent ledger writes lose nothing
    procs = [subprocess.Popen([sys.executable, f"{HERE}/ledger.py", "note", f"S.c{i}", "--result", f"r{i}"], env=env, stdout=subprocess.DEVNULL) for i in range(8)]
    assert all(p.wait() == 0 for p in procs)
    assert all(rows()[f"S.c{i}"]["result"] == f"r{i}" for i in range(8)), "a concurrent note was lost"
    # no git identity: refused before anything runs
    T2 = tempfile.mkdtemp(prefix="ar_selftest_noid_")
    os.makedirs(f"{T2}/code")
    json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", run_records="runs", code_dirs=["code"]), open(f"{T2}/project.json", "w"))
    subprocess.run(["git", "-c", "safe.directory=*", "-C", T2, "init", "-q"], check=True)
    noid = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.noid", "--hyp", "selftest", "--pred", "p", "--rule", "r", "--", "bash", "-c", "echo x"], capture_output=True, text=True,
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
    full = dict(schema=1, split="val", frames=10, commit="selftest", date="2026-10-25 17:00")  # the evaluator contract
    json.dump(dict(full, name="v", metrics=dict(Q=0.760, F=0.9), compare=dict(name="c", metrics=dict(Q=0.730, F=0.9), paired=dict(Q=dict(diff=0.03, lo=0.01, hi=0.05)))), open(f"{T}/runs/eval/v/val.json", "w"))
    json.dump(dict(full, name="c", metrics=dict(Q=0.730, F=0.9)), open(f"{T}/runs/eval/c/val.json", "w"))
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
    # a plain clause on subset-only arms judges every subset; lower-is-better by the experiment's direction; direction must match known names
    assert verdict(arms=subs, metric="R", rule=[dict(metric="R", min_effect=0.05)]) == "CUT", "plain clause on subset-only arms: all subsets needed"
    json.dump(dict(a=dict(err=0.10), b=dict(err=0.11), ca=dict(err=0.15), cb=dict(err=0.16)), open(f"{T}/err.json", "w"))
    lower = dict(a=dict(cards=["err.json#a", "err.json#b"]), c=dict(cards=["err.json#ca", "err.json#cb"]))
    assert verdict(arms=lower, metric="err", direction="lower", rule=[dict(metric="err", min_effect=0.01)]) == "KEEP", "lower is better by direction"
    assert verdict(arms=lower, metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)]) == "CUT", "the same numbers read as higher-is-better"
    import experiment as X
    assert any("lower-is-better" in b for b in X.check(dict(base, arms=two, metric="vert100", direction="higher", rule=[dict(metric="vert100", min_effect=0.1)]), f"{T}/experiments/E1_x/experiment.json"))
    exp = dict(base, arms=two)
    json.dump(exp, open(f"{T}/experiments/E1_x/experiment.json", "w"))
    assert X.check(exp, f"{T}/experiments/E1_x/experiment.json") == []
    assert any("control" in b for b in X.check(dict(exp, control="z"), f"{T}/experiments/E1_x/experiment.json"))
    assert any("no metric" in b for b in X.check(dict(exp, rule=[dict(metric="Q", min_effect=0.01), dict(metric="nope", max_loss=0.1)]), f"{T}/experiments/E1_x/experiment.json"))
    assert subprocess.run([sys.executable, f"{HERE}/decide.py", "--experiment", f"{T}/experiments/E1_x/experiment.json", "--write"], capture_output=True, text=True, env=env).returncode == 0
    assert os.path.exists(f"{T}/experiments/E1_x/decision.json")
    # tick: item states are derived from the run records and never written back; start what fits (gpu slots = the launchers' own slot directories), commands from the schedule or the experiment arm
    import schedule as SC
    for rid in ("S.ta", "S.tb"):
        subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", rid, "--hyp", "tick", "--pred", "p", "--rule", "r"], check=True, env=env, capture_output=True)
    exp = json.load(open(f"{T}/experiments/E1_x/experiment.json"))
    exp["arms"]["a"]["command"] = "bash -c 'echo arm a'"
    json.dump(exp, open(f"{T}/experiments/E1_x/experiment.json", "w"))
    SC.save([dict(item="S.ta", kind="train", state="planned", resource="cpu", priority="1", command="bash -c 'sleep 1; echo a'"),
             dict(item="S.tb", kind="train", state="planned", depends_on="S.ta", resource="gpu", priority="1", budget="5m", command="bash -c 'echo b'"),
             dict(item="E1.a", kind="train", state="planned", resource="cpu", priority="2", run_ids="E1.a"),
             dict(item="S.sum", kind="summary", state="planned", resource="agent", priority="3"),
             dict(item="S.nocmd", kind="train", state="planned", resource="cpu", priority="4"),
             dict(item="S.nocard", kind="train", state="planned", resource="cpu", priority="5", command="bash -c 'echo x'")], f"{T}/schedule.csv")
    raw = lambda: {r["item"]: r for r in SC.load(f"{T}/schedule.csv")}  # noqa: E731  (what the file says)
    shown = lambda: {r["item"]: r for r in SC.derived(SC.load(f"{T}/schedule.csv"), f"{T}/runs")}  # noqa: E731  (what schedule.py shows)

    def tick(*extra):
        return subprocess.run([sys.executable, f"{HERE}/tick.py", *extra], capture_output=True, text=True, env=env)

    def wait_done(*ids):
        for _ in range(300):
            if all(RUN.state_of(f"{T}/runs/{i}") == "done" for i in ids):
                return True
            time.sleep(0.2)
        return False
    dry = tick("--dry-run")
    assert "would start  S.ta" in dry.stdout and "would start  E1.a" in dry.stdout and raw()["S.ta"]["state"] == "planned" and not os.path.exists(f"{T}/runs/S.ta"), dry.stdout
    t1 = tick()
    assert "started  S.ta" in t1.stdout and "started  E1.a" in t1.stdout and "S.tb" not in t1.stdout, t1.stdout + t1.stderr  # S.tb waits for S.ta: not ready, not listed
    assert "command  S.nocmd: no command" in t1.stdout and "lead     S.sum" in t1.stdout, t1.stdout
    assert raw()["S.nocard"]["state"] == "blocked" and "launcher exit 2" in raw()["S.nocard"]["note"] and t1.returncode == 1, (raw()["S.nocard"], t1.stdout)
    assert wait_done("S.ta", "E1.a"), open(f"{T}/runs/S.ta/launcher.log").read()
    assert RUN.record(f"{T}/runs/E1.a")["hypothesis"] == exp["title"], "the arm's card reaches the run record"
    assert shown()["S.ta"]["state"] == "done" and raw()["S.ta"]["state"] == "planned", "the state is derived from the record; the file is not written back"
    t2 = tick()
    assert "started  S.tb" in t2.stdout, t2.stdout + t2.stderr
    assert wait_done("S.tb") and "t/S.tb" in tags(), "the gpu item ran with a slot and left its tag"
    assert shown()["S.tb"]["state"] == "done" and raw()["S.tb"]["state"] == "planned"
    t3 = tick()
    assert "started" not in t3.stdout and "done     3" in t3.stdout, t3.stdout  # nothing is started twice
    # GPU slots: the launchers' own slot directories are the count; a stale one does not count; a gpu item waits when none is free and runs when one is
    for i in (0, 1):
        os.makedirs(f"{T}/runs/S.hold{i}", exist_ok=True)
        json.dump(dict(id=f"S.hold{i}", status="running", launcher_pid=1), open(f"{T}/runs/S.hold{i}/run.json", "w"))
        open(f"{T}/runs/S.hold{i}/heartbeat", "w").write("x")
        os.makedirs(f"{T}/runs/.gpu_slots/{i}", exist_ok=True)
        json.dump(dict(id=f"S.hold{i}", pid=os.getpid(), run_dir=f"{T}/runs/S.hold{i}"), open(f"{T}/runs/.gpu_slots/{i}/run.json", "w"))
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", "S.tg", "--hyp", "tick", "--pred", "p", "--rule", "r"], check=True, env=env, capture_output=True)
    SC.save([dict(item="S.tg", kind="train", state="planned", resource="gpu", priority="1", command="bash -c 'echo g'")], f"{T}/schedule.csv")
    full_slots = tick()
    assert "waiting  S.tg: no free GPU slot" in full_slots.stdout and "gpu free 0" in full_slots.stdout, full_slots.stdout
    shutil.rmtree(f"{T}/runs/.gpu_slots/1")
    free_slot = tick()
    assert "started  S.tg" in free_slot.stdout, free_slot.stdout
    assert wait_done("S.tg")
    shutil.rmtree(f"{T}/runs/.gpu_slots")
    import tick as TK
    assert TK.budget_minutes("40m") == 40 and TK.budget_minutes("1.5 h") == 90 and TK.budget_minutes("90") == 90 and TK.budget_minutes("2 d") == 2880 and TK.budget_minutes("soon") is None
    # the evaluator contract: a benchmark script (a code system, no ML) writes cards; run.py runs it from planned rows; card.py checks them; decide reads them
    import card as CARD
    CARD.selftest()
    open(f"{T}/bench.py", "w").write(
        "import json, os, sys, time\n"
        "name, ms = sys.argv[1], float(sys.argv[2])\n"
        "os.makedirs(f'runs/eval/{name}', exist_ok=True)\n"
        "json.dump(dict(schema=1, name=name, split='bench', n=5, metrics=dict(ms=ms, pass_rate=1.0), commit='selftest', date=time.strftime('%Y-%m-%d %H:%M')), open(f'runs/eval/{name}/bench.json', 'w'))\n")
    for rid, nm, ms in (("E2.a", "fastlib", "8.0"), ("E2.c", "baselib", "12.0")):
        subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", rid, "--hyp", "the new library is faster", "--pred", "ms -4 vs c", "--rule", "keep if ms lower by 1"], check=True, env=env, capture_output=True)
        b = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", rid, "--no-git", "--split", "bench", "--", sys.executable, f"{T}/bench.py", nm, ms], capture_output=True, text=True, env=env, cwd=T)
        assert b.returncode == 0, b.stdout + b.stderr
    chk = subprocess.run([sys.executable, f"{HERE}/card.py", "check", "--all"], capture_output=True, text=True, env=env)
    assert chk.returncode == 0 and chk.stdout.count("ok   ") >= 2, chk.stdout
    os.makedirs(f"{T}/experiments/E2_bench", exist_ok=True)
    json.dump(dict(id="E2", title="faster library", kind="variant", metric="ms", direction="lower", band=0.5, prediction=dict(value=-4, vs="c"), control="c", deterministic=True,
                   arms=dict(a=dict(names=["fastlib"]), c=dict(names=["baselib"])), rule=[dict(metric="ms", min_effect=1.0), dict(metric="pass_rate", max_loss=0.0)],
                   evaluate=dict(split="bench", compare="c")), open(f"{T}/experiments/E2_bench/experiment.json", "w"))
    assert X.check(json.load(open(f"{T}/experiments/E2_bench/experiment.json")), f"{T}/experiments/E2_bench/experiment.json") == []
    d2 = DC.decide_experiment(json.load(open(f"{T}/experiments/E2_bench/experiment.json")), T)
    assert d2["verdict"] == "INCONCLUSIVE" and d2["effect"] == 4.0, d2  # one benchmark run each, no interval: a repeat is needed - the same rule as for seeds
    bad_card = dict(schema=1, name="x", split="bench", n=5, metrics=dict(ms="slow"), commit="c", date="d")
    json.dump(bad_card, open(f"{T}/runs/eval/fastlib/bad.json", "w"))
    assert subprocess.run([sys.executable, f"{HERE}/card.py", "check", f"{T}/runs/eval/fastlib/bad.json"], capture_output=True, text=True, env=env).returncode == 1
    os.remove(f"{T}/runs/eval/fastlib/bad.json")
    # the holdout split is a project setting: 'bench' is free here, the holdout needs the logged id
    json.dump(dict(json.load(open(f"{T}/project.json")), holdout_split="final"), open(f"{T}/project.json", "w"))
    assert subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.final", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--split", "final", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env).returncode == 2
    assert subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.test2", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--split", "test", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env).returncode == 0, "test is no longer the holdout here"
    open(f"{T}/TEST.md", "a").write("| S.final.ok | logged |\n")
    hold = lambda rid, split: subprocess.run([sys.executable, f"{HERE}/run.py", "--id", rid, "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--split", split, "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=env)  # noqa: E731
    assert hold("S.final.ok", "final").returncode == 0, "the renamed holdout with a logged id runs"
    assert hold("S.final2", "Final").returncode == 2 and hold("S.final3", " FINAL ").returncode == 2, "spelling does not bypass the holdout guard"
    json.dump(dict(json.load(open(f"{T}/project.json")), splits=["val", "final"]), open(f"{T}/project.json", "w"))
    assert "not one of the project's splits" in hold("S.other", "other").stdout
    json.dump({k: v for k, v in json.load(open(f"{T}/project.json")).items() if k not in ("holdout_split", "splits")}, open(f"{T}/project.json", "w"))
    # a legacy card (no schema field) still resolves in an experiment; card.py reports it as LEGACY, --strict as broken; the cards folder is a project setting
    os.makedirs(f"{T}/runs/eval/leg", exist_ok=True)
    json.dump(dict(name="leg", split="val", frames=10, metrics=dict(Q=0.75, F=0.9), commit="old", date="2026-10-20"), open(f"{T}/runs/eval/leg/val.json", "w"))
    exp3 = dict(base, id="E3", arms=dict(a=dict(names=["leg"]), c=dict(names=["c"])))
    os.makedirs(f"{T}/experiments/E3_leg", exist_ok=True)
    json.dump(exp3, open(f"{T}/experiments/E3_leg/experiment.json", "w"))
    assert X.check(exp3, f"{T}/experiments/E3_leg/experiment.json") == []
    lg = subprocess.run([sys.executable, f"{HERE}/card.py", "check", f"{T}/runs/eval/leg/val.json"], capture_output=True, text=True, env=env)
    assert lg.returncode == 0 and "LEGACY" in lg.stdout, lg.stdout
    assert subprocess.run([sys.executable, f"{HERE}/card.py", "check", "--strict", f"{T}/runs/eval/leg/val.json"], capture_output=True, text=True, env=env).returncode == 1
    os.makedirs(f"{T}/bench/cards/x", exist_ok=True)
    os.makedirs(f"{T}/bench/cards/y", exist_ok=True)
    for nm, q in (("x", 0.8), ("y", 0.7)):
        json.dump(dict(full, name=nm, metrics=dict(Q=q, F=0.9)), open(f"{T}/bench/cards/{nm}/val.json", "w"))
    json.dump(dict(json.load(open(f"{T}/project.json")), cards="bench/cards"), open(f"{T}/project.json", "w"))
    os.environ["AR_PROJECT"] = T  # X.check reads the project's card folder in this process
    try:
        assert X.check(dict(base, id="E4", arms=dict(a=dict(names=["x"]), c=dict(names=["y"]))), f"{T}/experiments/E3_leg/experiment.json") == [], "names resolve under the project's card folder"
    finally:
        del os.environ["AR_PROJECT"]
    json.dump({k: v for k, v in json.load(open(f"{T}/project.json")).items() if k != "cards"}, open(f"{T}/project.json", "w"))
    # one truth for a run in progress: the run record. A hand-started launcher is visible in schedule.py show and ledger.py tail without any sync; its ledger row appears when it ends
    SC.save([dict(item="S.live", kind="train", state="planned", resource="cpu", priority="1", run_ids="S.live2")], f"{T}/schedule.csv")
    live = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.live2", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "sleep 4; echo x"], env=env, stdout=subprocess.DEVNULL)
    for _ in range(100):
        if RUN.record(f"{T}/runs/S.live2").get("status") == "running":
            break
        time.sleep(0.1)
    time.sleep(0.5)
    assert shown()["S.live"]["state"] == "running" and raw()["S.live"]["state"] == "planned", "the schedule shows a hand-started launcher; the file is untouched"
    lt = subprocess.run([sys.executable, f"{HERE}/ledger.py", "tail"], capture_output=True, text=True, env=env)
    assert "in progress" in lt.stdout and "S.live2" in lt.stdout and "S.live2" not in rows(), lt.stdout
    assert live.wait() == 0
    assert rows()["S.live2"]["status"] == "ok" and rows()["S.live2"]["exit"] == "0" and shown()["S.live"]["state"] == "done" and raw()["S.live"]["state"] == "planned"
    # ledger sync: a finished run without a row gets one, a planned row is completed, a row that was judged or stopped by hand is left alone
    for name, status in (("S.fin_a", "ok"), ("S.fin_b", "failed"), ("S.fin_c", "ok")):
        os.makedirs(f"{T}/runs/{name}", exist_ok=True)
        json.dump(dict(id=name, status=status, exit=0 if status == "ok" else 1, duration_s=42, started="2026-10-25T10:00:00", ended="2026-10-25T10:01:00", hypothesis="h", command="c", launcher_pid=1),
                  open(f"{T}/runs/{name}/run.json", "w"))
        if status == "ok":
            open(f"{T}/runs/{name}/done", "w").write("x")
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", "S.fin_b", "--hyp", "planned card", "--pred", "p", "--rule", "r"], check=True, env=env, capture_output=True)
    RUN.ledger_write(f"{T}/LEDGER.csv", dict(id="S.fin_c", status="stopped", started="2026-10-25T10:00:00", notes="stopped by hand"))
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync"], capture_output=True, text=True, env=env, check=True)
    assert rows()["S.fin_a"]["status"] == "ok" and rows()["S.fin_a"]["duration_s"] == "42", rows()["S.fin_a"]
    assert rows()["S.fin_b"]["status"] == "failed" and rows()["S.fin_b"]["prediction"] == "p", rows()["S.fin_b"]
    assert rows()["S.fin_c"]["status"] == "stopped" and rows()["S.fin_c"]["notes"] == "stopped by hand", rows()["S.fin_c"]
    # decide --write puts a verdict on existing ledger rows only: an id that never ran gets no empty 'planned' row
    json.dump(dict(full, name="v", metrics=dict(Q=0.760, F=0.9)), open(f"{T}/runs/eval/v/val.json", "w"))
    json.dump(dict(base, id="E6", arms=dict(a=dict(names=["v"], run_ids=["S.live2", "S.never_ran"]), c=dict(names=["c"])), deterministic=True),
              open(f"{T}/experiments/E1_x/experiment.json", "w"))
    before = set(rows())
    wr = subprocess.run([sys.executable, f"{HERE}/decide.py", "--experiment", f"{T}/experiments/E1_x/experiment.json", "--write"], capture_output=True, text=True, env=env, cwd=T)
    assert wr.returncode == 0 and "no ledger row S.never_ran" in wr.stdout and set(rows()) == before, wr.stdout + wr.stderr
    assert "decide.py" in rows()["S.live2"]["decision"], rows()["S.live2"]
    pw = subprocess.run([sys.executable, f"{HERE}/decide.py", "--variant", "0.76,0.75", "--control", "0.73,0.72", "--id", "S.never_plain", "--write"], capture_output=True, text=True, env=env, cwd=T)
    assert pw.returncode == 0 and "no ledger row S.never_plain" in pw.stdout and "S.never_plain" not in rows(), "the typed-score path makes no ghost row either"
    # a command that cannot even be started is a failed run (exit 127), not a launcher that dies
    nb = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.nobin", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "/nonexistent/binary"], capture_output=True, text=True, env=env)
    assert nb.returncode == 127 and rows()["S.nobin"]["status"] == "failed" and RUN.record(f"{T}/runs/S.nobin")["status"] == "failed", (nb.stdout, nb.stderr[-300:])
    # a re-run of an id that has a ledger row starts a new judgement: result and decision of the earlier attempt are gone at its end
    RUN.ledger_write(f"{T}/LEDGER.csv", dict(id="S.old", status="ok", started="2026-10-24T10:00:00", result="old result", decision="KEEP", exit="0", duration_s="5", hypothesis="o"))
    assert go("S.old", ["bash", "-c", "echo x"]).returncode == 0
    assert rows()["S.old"]["status"] == "ok" and rows()["S.old"]["result"] == "" and rows()["S.old"]["decision"] == "", rows()["S.old"]
    # a busy schedule file does not matter to a launcher: run.py never touches it
    open(f"{T}/schedule.csv.lock", "w").write("held")
    t0 = time.time()
    slow = subprocess.run([sys.executable, f"{HERE}/run.py", "--id", "S.nolag", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "echo x"], capture_output=True, text=True, env=dict(env, AR_LOCK_WAIT_S="60"))
    os.remove(f"{T}/schedule.csv.lock")
    assert slow.returncode == 0 and time.time() - t0 < 15, (time.time() - t0, slow.stdout)
    # the schedule CLI loads and saves under the lock; a cancelled item stays cancelled whatever its run record says, any other stored state is only a note for an item with a record
    assert subprocess.run([sys.executable, f"{HERE}/schedule.py", "set", "S.live", "cancelled"], capture_output=True, text=True, env=env).returncode == 0 and raw()["S.live"]["state"] == "cancelled"
    assert shown()["S.live"]["state"] == "cancelled", "cancelled wins over a done record"
    sp = subprocess.run([sys.executable, f"{HERE}/schedule.py", "set", "S.live", "planned"], capture_output=True, text=True, env=env)
    assert "is shown as done" in sp.stdout and shown()["S.live"]["state"] == "done", sp.stdout
    # a launcher that died (heartbeat gone, no final write) is a failed item and, after sync, a counted `stalled` row; sync is idempotent and never replaces a finished row
    os.makedirs(f"{T}/runs/S.dead", exist_ok=True)
    json.dump(dict(id="S.dead", status="running", created="2026-10-25T10:00:00", command="c", launcher_pid=1), open(f"{T}/runs/S.dead/run.json", "w"))
    open(f"{T}/runs/S.dead/heartbeat", "w").write("x")
    os.utime(f"{T}/runs/S.dead/heartbeat", (time.time() - 3600, time.time() - 3600))
    assert SC.derived([dict(item="S.deadi", state="planned", run_ids="S.dead", resource="gpu")], f"{T}/runs")[0]["state"] == "failed", "a stalled record is a failed item"
    lt = subprocess.run([sys.executable, f"{HERE}/ledger.py", "tail"], capture_output=True, text=True, env=env)
    assert "without a ledger row" in lt.stdout and "S.dead" in lt.stdout, lt.stdout
    sy = subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync"], capture_output=True, text=True, env=env)
    assert rows()["S.dead"]["status"] == "stalled" and "launcher gone" in rows()["S.dead"]["notes"] and "1 row(s) added" in sy.stdout, (rows()["S.dead"], sy.stdout)
    assert "0 row(s) added" in subprocess.run([sys.executable, f"{HERE}/ledger.py", "sync"], capture_output=True, text=True, env=env).stdout, "a second sync adds nothing"
    assert RUN.ledger_write(f"{T}/LEDGER.csv", dict(id="S.ok", status="failed"), fill=True) is False and rows()["S.ok"]["status"] == "ok", "a finished row is not replaced by a fill"
    # tick: an item `A;B` whose A is done starts B (not A again); a dry run writes nothing, also when an item cannot be started; a real run marks it blocked
    for rid in ("S.m2", "S.c1", "S.c2", "S.race"):
        subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", rid, "--hyp", "t", "--pred", "p", "--rule", "r"], check=True, env=env, capture_output=True)
    os.makedirs(f"{T}/runs/S.m1", exist_ok=True)
    json.dump(dict(id="S.m1", status="ok", exit=0, launcher_pid=1), open(f"{T}/runs/S.m1/run.json", "w"))
    open(f"{T}/runs/S.m1/done", "w").write("x")
    subprocess.run([sys.executable, f"{HERE}/ledger.py", "add", "S.bq", "--hyp", "t", "--pred", "p", "--rule", "r"], check=True, env=env, capture_output=True)
    SC.save([dict(item="S.multi", kind="train", state="planned", resource="cpu", priority="1", run_ids="S.m1;S.m2", command="bash -c 'echo m'"),
             dict(item="S.bq", kind="train", state="planned", resource="gpu", priority="2", command="bash -c 'echo")], f"{T}/schedule.csv")  # unbalanced quote: cannot be started
    dry = tick("--dry-run")
    assert "would start  S.multi -> S.m2" in dry.stdout and "blocked  S.bq" in dry.stdout and raw()["S.bq"]["state"] == "planned" and dry.returncode == 1, dry.stdout
    real = tick()
    assert "started  S.multi -> S.m2" in real.stdout and raw()["S.bq"]["state"] == "blocked" and wait_done("S.m2"), real.stdout
    # cpu items run one at a time (cpu_slots, default 1); two ticks at once start an item once; two launchers of one id: the second is refused
    json.dump({k: v for k, v in json.load(open(f"{T}/project.json")).items() if k != "cpu_slots"}, open(f"{T}/project.json", "w"))
    SC.save([dict(item="S.c1", kind="train", state="planned", resource="cpu", priority="1", command="bash -c 'sleep 3; echo 1'"),
             dict(item="S.c2", kind="train", state="planned", resource="cpu", priority="2", command="bash -c 'echo 2'")], f"{T}/schedule.csv")
    c1 = tick()
    assert "started  S.c1" in c1.stdout and "waiting  S.c2: cpu slot busy" in c1.stdout, c1.stdout
    assert "waiting  S.c2: cpu slot busy" in tick().stdout, "a running cpu item holds the slot (derived from its record)"
    assert wait_done("S.c1") and "started  S.c2" in tick().stdout and wait_done("S.c2")
    SC.save([dict(item="S.race", kind="train", state="planned", resource="cpu", priority="1", command="bash -c 'sleep 3; echo r'")], f"{T}/schedule.csv")
    ticks = [subprocess.Popen([sys.executable, f"{HERE}/tick.py"], stdout=subprocess.PIPE, text=True, env=env) for _ in range(2)]
    outs = [p.communicate()[0] for p in ticks]
    assert sum("started  S.race" in o for o in outs) == 1, outs
    assert wait_done("S.race") and open(f"{T}/runs/S.race/run.log").read().count("=== attempt") == 1, "one command ran, not two"
    dup = [subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.dup", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "sleep 3; echo x"], env=env, stdout=subprocess.DEVNULL) for _ in range(2)]
    assert sorted(p.wait() for p in dup) == [0, 2], "exactly one launcher of an id runs"
    # a re-run drops the old judgement when it starts, not only when it ends (a re-run that dies must not keep the old KEEP)
    RUN.ledger_write(f"{T}/LEDGER.csv", dict(id="S.old2", status="ok", started="2026-10-24T10:00:00", result="old", decision="KEEP", exit="0", duration_s="5", hypothesis="o"))
    rr = subprocess.Popen([sys.executable, f"{HERE}/run.py", "--id", "S.old2", "--hyp", "h", "--pred", "p", "--rule", "r", "--no-git", "--", "bash", "-c", "sleep 3; echo x"], env=env, stdout=subprocess.DEVNULL)
    for _ in range(100):
        if RUN.record(f"{T}/runs/S.old2").get("status") == "running":
            break
        time.sleep(0.1)
    assert rows()["S.old2"]["decision"] == "" and rows()["S.old2"]["result"] == "", rows()["S.old2"]
    assert rr.wait() == 0
    # a chain waits long by default: --wait-after and --wait-gpu are 720 minutes
    src = open(f"{HERE}/run.py", encoding="utf-8").read()
    assert '"--wait-gpu", type=int, default=720' in src and '"--wait-after", type=int, default=720' in src
    # check_numbers
    open(f"{T}/doc.md", "w").write("S 0.746 and 0.99 here")
    open(f"{T}/src.json", "w").write('{"S": 0.74612}')
    out = subprocess.run([sys.executable, f"{HERE}/check_numbers.py", f"{T}/doc.md", f"{T}/src.json"], capture_output=True, text=True).stdout
    assert "2 decimal numbers, 1 found" in out and "0.99" in out, out
    # schedule: next respects dependencies, the state of an item with runs is derived from the run records
    import schedule as SC
    sch = [dict(item="a", state="planned", depends_on="", priority="1", run_ids="S.ok2", resource="gpu"),
           dict(item="b", state="planned", depends_on="a", priority="1", run_ids="S.oom", resource="gpu"),
           dict(item="c", state="planned", depends_on="", priority="2", run_ids="", resource="cpu")]
    assert [x["item"] for x in SC.ready(sch)] == ["a", "c"]
    st = {x["item"]: x["state"] for x in SC.derived(sch, f"{T}/runs")}
    assert st == {"a": "done", "b": "failed", "c": "planned"} and sch[0]["state"] == "planned", st  # derived from the run records; the input rows are untouched
    import clean_temp
    clean_temp.selftest()
    import guard_hook  # P4: protected files, grants, json entries, Windows spellings
    guard_hook.selftest()
    import metrics  # P3: loop table from ledger, decision files and TEST log
    metrics.selftest()
    strict = subprocess.run([sys.executable, f"{HERE}/check_numbers.py", f"{T}/doc.md", f"{T}/src.json", "--strict"], capture_output=True, text=True)
    assert strict.returncode == 1, "check_numbers --strict must fail on a number missing from the sources"
    print("autoresearch tools selftest passed")
finally:
    shutil.rmtree(T, ignore_errors=True)

"""Self-test of the autoresearch tools in a throw-away project with its own git repository (the real ledger is never touched).

    python autoresearch/tools/selftest.py        (needs git and bash, e.g. in the WSL)

Tests: run.py (done flag only on success, CUDA retry, card check, TEST guard, --after waits / blocks, commit + tag per run, a resumed id keeps
its tag, a planned ledger row keeps its prediction), ledger.py, decide.py, check_numbers.py, schedule.py (next, reconcile), clean_temp.py.
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
    again = go("S.ok", ["bash", "-c", "echo resume"])
    assert again.returncode == 0 and "tag existed" in rows()["S.ok"]["notes"]
    # --after: waits for done flags, blocks when a dependency failed
    aft = go("S.after", ["bash", "-c", "echo after"], "--after", "S.ok,S.ok2")
    blk = go("S.blocked", ["bash", "-c", "echo never"], "--after", "S.bad")
    assert aft.returncode == 0 and blk.returncode == 3 and rows()["S.blocked"]["status"] == "blocked"
    # card check and TEST guard
    assert go("S.card", ["bash", "-c", "echo x"], "--require-card").returncode == 2
    assert go("S.test", ["bash", "-c", "echo x"], "--split", "test").returncode == 2
    assert go("S.test.ok", ["bash", "-c", "echo x"], "--split", "test").returncode == 0
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
    # check_numbers
    open(f"{T}/doc.md", "w").write("S 0.746 and 0.99 here")
    open(f"{T}/src.json", "w").write('{"S": 0.74612}')
    out = subprocess.run([sys.executable, f"{HERE}/check_numbers.py", f"{T}/doc.md", f"{T}/src.json"], capture_output=True, text=True).stdout
    assert "2 decimal numbers, 1 found" in out and "0.99" in out, out
    # schedule: next respects dependencies, reconcile reads the run records
    import schedule as SC
    sch = [dict(item="a", state="planned", depends_on="", priority="1", run_ids="S.ok", resource="gpu"),
           dict(item="b", state="planned", depends_on="a", priority="1", run_ids="S.bad", resource="gpu"),
           dict(item="c", state="planned", depends_on="", priority="2", run_ids="", resource="cpu")]
    assert [x["item"] for x in SC.ready(sch)] == ["a", "c"]
    changed = SC.reconcile(sch, f"{T}/runs")
    assert ("a", "planned", "done") in changed and ("b", "planned", "failed") in changed, changed
    import clean_temp
    clean_temp.selftest()
    print("autoresearch tools selftest passed")
finally:
    shutil.rmtree(T, ignore_errors=True)

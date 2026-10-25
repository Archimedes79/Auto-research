"""Self-test of run.py / ledger.py: a successful command, a failing command (no done flag), a CUDA-error command that is retried, and the ledger render.  Cleans up after itself.
Run: scripts/_py14.sh ../tools/experiment/selftest.py"""
import csv
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(HERE, "run.py")
EXEC = os.path.join(ROOT, "experiment_execution")


def go(rid, cmd, extra=()):
    return subprocess.run([sys.executable, RUN, "--id", rid, "--hyp", "selftest", "--retries", "1", *extra, "--", *cmd], capture_output=True, text=True)


ok = go("SELFTEST.ok", ["bash", "-c", "echo hello"])
bad = go("SELFTEST.bad", ["bash", "-c", "echo boom; exit 3"])
cuda = go("SELFTEST.cuda", ["bash", "-c", "echo 'CUDA error: unknown error'; exit 1"])
print(ok.stdout.strip(), "|", bad.stdout.strip(), "|", cuda.stdout.strip(), sep="\n")
assert ok.returncode == 0 and os.path.exists(f"{EXEC}/runs/SELFTEST.ok/done")
assert bad.returncode == 3 and not os.path.exists(f"{EXEC}/runs/SELFTEST.bad/done")
assert cuda.returncode == 1 and "2 attempt" in cuda.stdout and not os.path.exists(f"{EXEC}/runs/SELFTEST.cuda/done")
rows = {r["id"]: r for r in csv.DictReader(open(f"{EXEC}/LEDGER.csv", encoding="utf-8"))}
assert rows["SELFTEST.ok"]["status"] == "ok" and rows["SELFTEST.bad"]["status"] == "failed"
subprocess.run([sys.executable, os.path.join(HERE, "ledger.py"), "note", "SELFTEST.ok", "--result", "S 0.5", "--decision", "keep"], check=True)
print(open(f"{EXEC}/LEDGER.md", encoding="utf-8").read().strip().splitlines()[-3:])
# clean up: runs and ledger rows of the self-test
for r in ("SELFTEST.ok", "SELFTEST.bad", "SELFTEST.cuda"):
    shutil.rmtree(f"{EXEC}/runs/{r}", ignore_errors=True)
rest = [r for r in csv.DictReader(open(f"{EXEC}/LEDGER.csv", encoding="utf-8")) if not r["id"].startswith("SELFTEST")]
with open(f"{EXEC}/LEDGER.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes"])
    w.writeheader()
    w.writerows(rest)
subprocess.run([sys.executable, os.path.join(HERE, "ledger.py"), "render"], check=True)
print("experiment tools selftest passed")

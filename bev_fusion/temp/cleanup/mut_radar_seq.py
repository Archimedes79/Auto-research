"""scratch: three planted mutants of the radar_seq perturbation change; each must make a selftest fail. Files are restored in a finally block."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EV = os.path.join(ROOT, "evaluation")
MUTANTS = [
    ("bev_adapter.py", '            item["radar_seq"] = [{k: v[:0] for k, v in r.items()} for r in item["radar_seq"]]\n', "            pass\n", ["bev_selftest.py", "--only", "bev_probe"]),
    ("bev_adapter.py", '            item["radar_seq"] = other["radar_seq"]\n', "            pass\n", ["bev_selftest.py", "--only", "bev_probe"]),
    ("bev_modules.py", '        out["radar_seq"] = [fn(r) for r in batch["radar_seq"]]\n', "        pass\n", ["bev_modules.py", "--selftest"]),
    ("bev_probe.py", 'for r in {id(r): r for r in [batch["radar"], *batch.get("radar_seq", [])]}.values():', 'for r in [batch["radar"]]:', ["bev_selftest.py", "--only", "bev_probe"]),
    ("bev_probe.py", 'for r in {id(r): r for r in [batch["radar"], *batch.get("radar_seq", [])]}.values():', 'for r in [batch["radar"], *batch.get("radar_seq", [])]:', ["bev_selftest.py", "--only", "bev_probe"]),
    ("bev_probe.py", '            batch["radar_seq"] = [{k: v[:0] for k, v in r.items()} for r in batch["radar_seq"]]\n', "            pass\n", ["bev_selftest.py", "--only", "bev_probe"]),
]
env = dict(os.environ, PYTHONPATH=os.path.dirname(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="2")
py = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
killed = 0
for fname, old, new, cmd in MUTANTS:
    path = os.path.join(EV, fname)
    src = open(path, encoding="utf-8", newline="").read()
    assert src.count(old) == 1, (fname, old[:50], src.count(old))
    try:
        open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new))
        r = subprocess.run(["nice", "-n", "15", py] + [os.path.join(EV, cmd[0])] + cmd[1:], cwd=ROOT, env=env, capture_output=True, text=True)
        dead = r.returncode != 0
        killed += dead
        print(("KILLED " if dead else "SURVIVED"), fname, "|", old.strip()[:70], "->", new.strip()[:40])
    finally:
        open(path, "w", encoding="utf-8", newline="").write(src)
print(f"{killed}/{len(MUTANTS)} mutants killed")
sys.exit(0 if killed == len(MUTANTS) else 1)

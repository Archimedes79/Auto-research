"""my rows in the two shared files, on top of the committed blobs (another chat has uncommitted rows in both): tests/README.md and tests/run_gates.sh"""
import os
import subprocess

REPO = "/mnt/c/Auto-research"
OUT = REPO + "/bev_fusion/temp/head_session_2026-11-01"


def head(p):
    return subprocess.run(["git", "show", "HEAD:" + p], cwd=REPO, capture_output=True, text=True, check=True).stdout


def work_line(p, marker):
    ls = [x for x in open(os.path.join(REPO, p), encoding="utf-8").read().split("\n") if x.startswith(marker)]
    assert len(ls) == 1, (p, marker, len(ls))
    return ls[0]


for path, marker, anchor, out in (
        ("bev_fusion/tests/README.md", "| `test_host_targets.py` |", "| `test_slots.py` |", "tests_readme_mine.md"),
        ("bev_fusion/tests/run_gates.sh", 'run "host targets"', 'run "startable"', "run_gates_mine.sh")):
    h = head(path)
    assert h.count(anchor) == 1, (path, anchor, h.count(anchor))
    new = h.replace(anchor, work_line(path, marker) + "\n" + anchor, 1)
    open(os.path.join(OUT, out), "w", encoding="utf-8", newline="\n").write(new)
    print(path, "->", out, "+1 line")

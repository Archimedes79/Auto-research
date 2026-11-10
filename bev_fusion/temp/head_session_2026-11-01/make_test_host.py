"""writes tests/test_host_targets.py: the oracle is the OLD build_targets, copied textually from the committed object_head.py"""
import subprocess

REPO = "/mnt/c/Auto-research"
old = subprocess.run(["git", "show", "HEAD:bev_fusion/training/object_head.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
a = old.index("def build_targets(boxes_batch, grid, groups=GROUPS, max_obj=64, min_radius=2):")
b = old.index("# ------------------------------------------------------------------------------------------------ losses")
oracle = old[a:b].rstrip() + "\n"
oracle = oracle.replace("def build_targets(", "def build_targets_old(")
HEAD = open(REPO + "/bev_fusion/temp/head_session_2026-11-01/test_host_head.txt", encoding="utf-8").read()
TAIL = open(REPO + "/bev_fusion/temp/head_session_2026-11-01/test_host_tail.txt", encoding="utf-8").read()
open(REPO + "/bev_fusion/tests/test_host_targets.py", "w", encoding="utf-8", newline="\n").write(HEAD + oracle + TAIL)
print("written")

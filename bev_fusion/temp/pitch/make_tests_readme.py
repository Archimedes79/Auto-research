"""tests/README.md for the code commit: the committed (HEAD) file plus ONLY the row of test_velocity_hurdle.py from the working copy (the working copy also holds the GT chat's changed row of
test_gt_cache_targets.py and the lead's row of the uncommitted test_bev_prototype.py, which are not part of this commit).
usage: python3 temp/pitch/make_tests_readme.py <out file>"""
import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
head = subprocess.run(["git", "-C", root, "show", "HEAD:bev_fusion/tests/README.md"], capture_output=True, text=True, check=True).stdout
work = open(root + "/bev_fusion/tests/README.md", encoding="utf-8").read()
row = [l for l in work.splitlines() if l.startswith("| `test_velocity_hurdle.py` |")]
assert len(row) == 1, "expected exactly one hurdle row in the working README"
lines = head.splitlines()
i = [k for k, l in enumerate(lines) if l.startswith("| `test_gt_cache_targets.py` |")]
assert len(i) == 1, "anchor row of test_gt_cache_targets.py not found in HEAD"
lines.insert(i[0] + 1, row[0])
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(lines) + ("\n" if head.endswith("\n") else ""))
print("hurdle row inserted after line", i[0] + 1)

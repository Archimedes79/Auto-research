"""Scratch: rebuild DECISIONS.md = the version of e5a13211 (has D68) + my D67 update, so the D68 row dropped by commit 9943a320 comes back."""
import subprocess

P = "bev_fusion_lane_center/experiment_execution/DECISIONS.md"
parent = subprocess.check_output(["git", "show", f"e5a13211:{P}"])
head = subprocess.check_output(["git", "show", f"HEAD:{P}"])
pre = b"the shutdown waits for the run records of their ids; MAIL: no; no push."
# the D67 row's tail in HEAD (with my update) and in the parent (without it), each up to the closing " |"
i = head.index(pre)
new = head[i:head.index(b" |\n", i) + 2]
k = parent.index(pre)
old = parent[k:parent.index(b" |\n", k) + 2]
assert head.count(pre) == 1 and parent.count(pre) == 1
assert b"Nein, anlassen" in new and b"Nein, anlassen" not in old, (new[-200:], old[-200:])
fixed = parent.replace(old, new)
open("bev_fusion/temp/cleanup/DECISIONS_fixed.md", "wb").write(fixed)
print("parent", len(parent), "head", len(head), "fixed", len(fixed))

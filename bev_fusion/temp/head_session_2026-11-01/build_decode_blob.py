"""object_head.py as it is committed + ONLY my decode hunk (the working file also carries another chat's uncommitted hunks in build_targets / head_loss).
usage (repo root, WSL): python3 build_decode_blob.py  -> writes temp/head_session_2026-11-01/object_head_mine.py"""
import os
import subprocess

REPO = "/mnt/c/Auto-research"
P = "bev_fusion/training/object_head.py"
HERE = os.path.join(REPO, "bev_fusion/temp/head_session_2026-11-01")
head = subprocess.run(["git", "show", "HEAD:" + P], cwd=REPO, capture_output=True, text=True, check=True).stdout
work = open(os.path.join(REPO, P), encoding="utf-8").read()
OPEN, CLOSE = "@torch.no_grad()\ndef decode(", "class TimeObjectHead(ObjectHead):"
# my version of the segment: from the cache variable (or the decorator) to the next class
w_a = work.index("_boxes_from_maps = None\n\n\n" + OPEN)
w_b = work.index(CLOSE)
mine = work[w_a:w_b]
h_a, h_b = head.index(OPEN), head.index(CLOSE)
assert head.count(OPEN) == 1 and head.count(CLOSE) == 1 and work.count(CLOSE) == 1
out = head[:h_a] + mine + head[h_b:]
open(os.path.join(HERE, "object_head_mine.py"), "w", encoding="utf-8", newline="\n").write(out)
print("lines: head", head.count("\n"), "mine", out.count("\n"), "work", work.count("\n"))

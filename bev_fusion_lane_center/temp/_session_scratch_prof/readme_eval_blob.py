import subprocess
path = "bev_fusion_lane_center/evaluation/README.md"
s = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, text=True, encoding="utf-8", check=True).stdout
a = "sweep --split [--grid perclass]` (the grid of decode variants"
assert s.count(a) == 1
s = s.replace(a, "sweep --split [--grid perclass] [--jobs N]` (N forked processes score the cells through `pmap.py`, the same rows in the same order; the grid of decode variants")
open("bev_fusion_lane_center/temp/_blob_eval_readme.txt", "w", encoding="utf-8", newline="\n").write(s)

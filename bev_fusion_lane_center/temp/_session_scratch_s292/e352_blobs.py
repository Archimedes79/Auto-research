import os
import subprocess

R = "/mnt/c/Auto-research"
P = "bev_fusion_lane_center/"
OUT = R + "/bev_fusion_lane_center/temp/_session_scratch_s292/blobs_e352"
os.makedirs(OUT, exist_ok=True)
head = subprocess.run(["git", "-C", R, "show", "HEAD:" + P + "evaluation/README.md"], capture_output=True, text=True, check=True).stdout.split("\n")
work = open(os.path.join(R, P, "evaluation/README.md"), encoding="utf-8", newline="").read().split("\n")
mine = [l for l in work if l.startswith("| `jprune_b.py`")]
assert len(mine) == 1 and not any(l.startswith("| `jprune_b.py`") for l in head)
i = work.index(mine[0])
anchor = work[i - 1]
j = head.index(anchor)
head[j + 1:j + 1] = mine
open(OUT + "/evaluation_README.md", "w", encoding="utf-8", newline="").write("\n".join(head))
print("ok", len(head), len(work))

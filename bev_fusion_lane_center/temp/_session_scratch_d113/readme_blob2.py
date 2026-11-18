import io
import subprocess

p = "evaluation/README.md"
cur = io.open(p, encoding="utf-8").read().split("\n")
mine = [l for l in cur if l.startswith("| `jpast_eval.py` |")]
assert len(mine) == 1
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/" + p], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
i = [k for k, l in enumerate(head) if l.startswith("| `jpast_eval.py` |")]
assert len(i) == 1
head[i[0]] = mine[0]
io.open("temp/_session_scratch_d113/eval_readme_mine2.md", "w", encoding="utf-8", newline="\n").write("\n".join(head))
print("ok")

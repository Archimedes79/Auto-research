"""evaluation/README.md blob = HEAD with the jtoken_eval row replaced by the coder's E375 row (another chat's uncommitted edit of the same row stays out)"""
import subprocess
path = "evaluation/README.md"
head = subprocess.run(["git", "show", f"HEAD:bev_fusion_lane_center/{path}"], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
new = open("temp/e375/e375_readme_row_jtoken_eval.md", encoding="utf-8").read().strip("\n")
assert new.startswith("| `jtoken_eval.py` |"), new[:40]
i = [k for k, l in enumerate(head) if l.startswith("| `jtoken_eval.py` |")]
assert len(i) == 1
head[i[0]] = new
open("temp/e375/blob_evaluation_README.md", "w", encoding="utf-8").write("\n".join(head))
print("ok", i[0] + 1)

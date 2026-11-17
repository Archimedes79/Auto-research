"""evaluation/README.md blob = HEAD + the --model sentence in MY jtoken_eval row (the lane chat's jpast_eval row stays out)"""
import subprocess
path = "evaluation/README.md"
head = subprocess.run(["git", "show", f"HEAD:bev_fusion_lane_center/{path}"], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
tag = "| `jtoken_eval.py` |"
i = [k for k, l in enumerate(head) if l.startswith(tag)]
assert len(i) == 1
row = head[i[0]]
assert row.endswith("|")
add = " `--model model_step<N>.pt` (dump / score / sweep / fixture --run / graphs): a snapshot instead of model.pt; every file name carries `_step<N>`, the outputs record the model, a dump of other weights is refused."
head[i[0]] = row[:-1].rstrip() + add + " |"
open("temp/e365/blob_evaluation_README_2.md", "w", encoding="utf-8").write("\n".join(head))
print("ok")

"""prints the E286 arm (j0 / j1 / j2) with the best mean kp_f1 (entry / exit F1 at 3 m, thr 0.3, distance only) over the seeds that were scored; a tie inside 0.01 goes to the cheaper arm"""
import json
import os
import sys

R = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
score = {}
for arm in ("j0", "j1", "j2"):
    v = []
    for s in (0, 1):
        p = os.path.join(R, "runs", f"e286_{arm}_s{s}", "jmaps_val.json")
        if os.path.exists(p):
            kp = json.load(open(p))["keypoints"]
            v.append((kp["kp_ours_tol3.0_noang_ent"]["f1"] + kp["kp_ours_tol3.0_noang_ext"]["f1"]) / 2)
    if v:
        score[arm] = sum(v) / len(v)
if not score:
    sys.exit("no scored E286 arm")
best = max(score, key=score.get)
for arm in ("j0", "j1", "j2"):
    if arm in score and score[best] - score[arm] <= 0.01:
        best = arm
        break
print(best, file=sys.stdout)
print(json.dumps(score), file=sys.stderr)

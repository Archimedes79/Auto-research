"""scratch: schedule notes after the B2 result (2026-11-09): only the note column of t22.B2 and t22.twin changes (states are read from the run records)"""
import csv
import io

path = "records/schedule.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
n = 0
for r in rows:
    if r["item"] == "t22.B2":
        r["note"] = ("RAN 2026-11-09 02:24-05:07 (after the BEV system's chain7, night window): CUT by the written rule - lane divider IoU@1 -0.060 and mAP of the fusion area -0.077 against "
                     "t21.pair4_s0, intervals below -m; static-cell velocity EPE equal to the control (0.2150 / 0.2172), moving-centre +0.038. Card experiments/t22_student_encoder, 'Stage B2 result'. "
                     "Was: stage B2 with student_frames 'all' (1500 steps, P4 chain, warm start t20.base_s0), control t21.pair4_s0; Mirko 2026-11-08: B2 first (before the R arm)")
        n += 1
    if r["item"] == "t22.twin":
        r["note"] = ("seed twin of stage B (student + control, about 5 h); B2 is cut, so the student route is still INCONCLUSIVE at stage B (one seed). Waits for Mirko, who chooses between this twin, "
                     "the R arm (ResNet-18; 30 min distillation first), a B3 with gradient through the older frame (new code) and stopping. The small network becomes the baseline of the image side only "
                     "when the card's margins are met on the twin (D62)")
        n += 1
assert n == 2, n
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
w.writeheader()
w.writerows(rows)
open(path, "w", newline="", encoding="utf-8").write(buf.getvalue())
print("rows", len(rows), "notes changed", n)

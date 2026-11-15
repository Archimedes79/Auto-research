"""scratch: the schedule row t22.B2 gets the new arming (after the reboot of 2026-11-08 22:09, behind the BEV system's chain7); only the note of this row changes"""
import csv
import io

path = "records/schedule.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
for r in rows:
    if r["item"] == "t22.B2":
        r["note"] = ("stage B2 with student_frames 'all' (1500 steps, P4 chain, warm start t20.base_s0), control t21.pair4_s0; Mirko 2026-11-08: B2 first (before the R arm). "
                     "The reboot of 2026-11-08 22:09 killed the first waiter before anything started; re-armed 22:16 by temp/t22/rearm_B2_after_chain7.sh "
                     "behind the BEV system's t23.seed1 chain (flag temp/lane_tests/DONE7), then smoke, the run (latest start about 03:30), evaluation; night window only")
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
w.writeheader()
w.writerows(rows)
open(path, "w", newline="", encoding="utf-8").write(buf.getvalue())
print("rows", len(rows))

"""scratch: the schedule row t22.B2 gets the run id of the launcher (t22.B2_s0) and the state planned (Mirko 2026-11-08: B2 first); only this row changes"""
import csv
import io

path = "records/schedule.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
for r in rows:
    if r["item"] == "t22.B2":
        r["state"] = "planned"
        r["run_ids"] = "t22.B2_s0"
        r["note"] = ("stage B2 with student_frames 'all' (1500 steps, P4 chain, warm start t20.base_s0), control t21.pair4_s0; Mirko 2026-11-08: B2 first; "
                     "armed for the night window by temp/t22/night_B2.sh (smoke, then the run; waits for t23.lane_bce or 01:00, GPU free, latest start 03:30)")
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
w.writeheader()
w.writerows(rows)
open(path, "w", newline="", encoding="utf-8").write(buf.getvalue())
print("rows", len(rows))

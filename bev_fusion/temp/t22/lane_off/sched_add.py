"""scratch: add the schedule row t20.L4check (D65); only this row is appended, the rest of records/schedule.csv is rewritten as it was"""
import csv
import io

path = "records/schedule.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
assert not any(r["item"] == "t20.L4check" for r in rows)
row = {c: "" for c in cols}
row.update({"item": "t20.L4check", "node": "t20", "kind": "eval", "state": "planned", "depends_on": "t22.B2", "resource": "gpu", "budget": "0.4h", "priority": "3",
            "owner": "Image Encoder chat", "run_ids": "diag.t20_L4on;diag.t20_L4off",
            "note": ("D65 (Mirko 2026-11-08: 'remove it', 'cheap check first, then off'): the checkpoint of t20.base_s0 under the live code, VAL 1000, twice: the twin (lane branch on) and the lane branch "
                     "dropped after the load (not retrained); diagnostic, no card, no verdict. Waiter temp/t22/lane_off/waiter_lane_off.sh starts it after my B2 chain (rearm_B2_after_chain7.sh) has ended "
                     "and the BEV chain7 is over, night window only, GPU free on two polls. Then the off config training/configs/t20_base_nol4.json is the base for a new base run (waits for his word)")})
rows.append(row)
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
w.writeheader()
w.writerows(rows)
open(path, "w", newline="", encoding="utf-8").write(buf.getvalue())
print("rows", len(rows))

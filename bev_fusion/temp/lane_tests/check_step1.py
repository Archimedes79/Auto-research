"""scratch (t23 matched-control check, card section 1): step 1 of a new run must equal the control's row 1 in every log term the line BCE cannot touch.
python temp/lane_tests/check_step1.py runs/<new run>/log.jsonl [runs/t21_ctl_s0/log.jsonl]   -> exit 0 (equal), 1 (differs), 2 (no step-1 row)"""
import json
import sys

KEYS = ["group.areas", "group.objects", "group.motion", "objhead.vel@+0.0", "objhead.mov@+0.0", "motion@+0.0", "radar_aux"]
REL = 1e-4


def row1(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("step") == 1:
            return r
    return None


new, ctl = row1(sys.argv[1]), row1(sys.argv[2] if len(sys.argv) > 2 else "runs/t21_ctl_s0/log.jsonl")
if new is None or ctl is None:
    print("no step-1 row:", "new" if new is None else "", "control" if ctl is None else "")
    sys.exit(2)
bad = 0
for k in KEYS:
    a, b = new.get(k), ctl.get(k)
    ok = a is not None and b is not None and abs(a - b) <= REL * max(abs(b), 1e-12)
    print(f"  {'ok  ' if ok else 'DIFF'} {k:20s} new {a}  control {b}")
    bad += 0 if ok else 1
print("group.lines (expected to differ with the BCE):", new.get("group.lines"), "control", ctl.get("group.lines"))
print("step-1 check:", "equal" if not bad else f"{bad} term(s) differ")
sys.exit(1 if bad else 0)

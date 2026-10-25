"""One-off: correct the E234 run records after the extraction (D7 step 4).

    wsl python3 temp/loop19/fix_e234_records.py

E234.tr0 / tr1 started while the WSL git had no identity, so run.py recorded commit git-error; both were tagged by hand at 07a4bad (D9).
Sets commit / tag / git_note in their run.json, the commit column, the split (b456, not the launcher default val) and result / decision / notes
of all three rows (numbers from temp/loop19/check_b456_features.py), under the ledger lock; then renders LEDGER.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "autoresearch", "tools"))
import project as P  # noqa: E402
import ledger as L  # noqa: E402

TRAIN = "post-check PASS (train part, 8731 rows): all rows written, finite, std ratio vs b123 dec32 0.982, logit7 0.975, fused32 0.943"
VAL = "post-check PASS (val part = CONFIRM, 1567 rows): all rows written, finite, std ratio vs b123 dec32 0.945, logit7 1.041, fused32 0.982"
HAND = "WSL git had no identity at the start (commit git-error); tagged by hand at 07a4bad (D9)"
FIX = {
    "E234.tr0": dict(result=TRAIN, notes=HAND, commit="07a4badade", tag="lcr/E234.tr0"),
    "E234.tr1": dict(result=TRAIN, notes=HAND, commit="07a4badade", tag="lcr/E234.tr1"),
    "E234.val": dict(result=VAL, notes="", commit=None, tag=None),
}

runs = os.path.join(os.path.dirname(P.path("ledger")), "runs")
for rid, f in FIX.items():
    if f["commit"]:
        p = os.path.join(runs, rid, "run.json")
        with P.locked(p):
            r = json.load(open(p, encoding="utf-8"))
            r.update(commit=f["commit"], tag=f["tag"], git_note=HAND)
            P.atomic_write(p, json.dumps(r, indent=1) + "\n")
with P.locked(P.path("ledger")):
    rows = L.load()
    for r in rows:
        f = FIX.get(r["id"])
        if f:
            r.update(result=f["result"], decision="PASS: b456 usable for TRAIN2 / CONFIRM / UNSEEN (D7)", split="b456")
            if f["notes"]:
                r["notes"] = f["notes"]
            if f["commit"]:
                r["commit"] = f["commit"]
    L.save(rows)
L.render()

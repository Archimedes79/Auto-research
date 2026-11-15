#!/bin/bash
# scratch: result + decision of the two L4 diagnostic rows, and the note of the schedule row t20.L4check (only that note changes)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
L=../autoresearch/tools/ledger.py

python3 $L note diag.t20_L4on \
  --result 'D65 check, the twin: runs/t20_base_s0 checkpoint (step 6000) under the live code, lane branch ON, VAL 1000 frames, 82 scenes; reproduces the stored t20.base_s0 numbers to <= 0.0001 on every compared row (lane divider IoU@0.5 0.4257, mAP detf 0.4092, moving-centre EPE 1.1037)' \
  --decision 'no verdict (diagnostic, --no-card); harness check passed, read together with diag.t20_L4off' \
  --notes 'evaluator hash a38635126e2f against the stored e084cbc548a7: no number differs, so the warning of bev_suite compare is harmless here'

python3 $L note diag.t20_L4off \
  --result 'D65 check: the same checkpoint with the lane branch dropped after the load (248,064 parameters out of the forward; NOT retrained), VAL 1000, paired scene bootstrap b-a (b = dropped, a = on): lane divider IoU@0.5 0.4257->0.4240 -0.0017 [-0.0041,+0.0007], IoU@1 -0.0007 [-0.0032,+0.0018] (ties); lane divider zero-crossing F1 +0.0158 [+0.0068,+0.0244]; drivable +0.0005; velocity rows identical (1.1037 / 0.2282 / 0.6413); walkway IoU@0 -0.0081 [-0.0109,-0.0054]; road divider IoU@1 -0.0059 [-0.0111,-0.0010]; mAP fusion area -0.0102 [-0.0159,-0.0042], plain mAP -0.0090 [-0.0138,-0.0035]; pedestrian AP@1 -0.0169 [-0.0288,-0.0040]; val loss +0.083 [+0.064,+0.103]' \
  --decision 'no verdict (diagnostic, --no-card). Reading: nothing measurable on the lane lines (on minus dropped at most +0.004 at the 95 pct limit; the BEV system 246-frame read-out gave +0.0044 [-0.0001,+0.0090]); the 0.01 loss on boxes and walkway is the change of the camera features at test time in a network trained with the branch, NOT proof that a network trained without it is worse: only a retrain (about 7 h, config t20_base_nol4) answers that. D65 row updated'

python3 - <<'PYEOF'
import csv, io
path = "records/schedule.csv"
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
cols = list(rows[0].keys())
n = 0
for r in rows:
    if r["item"] == "t20.L4check":
        r["note"] = ("RAN 2026-11-09 05:24-05:44 (diag.t20_L4on 574 s, diag.t20_L4off 524 s): the twin reproduces the stored t20.base_s0 to <= 0.0001; dropping L4 leaves lane divider IoU@0.5 / @1 unchanged "
                     "(-0.0017 [-0.0041,+0.0007] / -0.0007), velocity identical, but walkway -0.0081, mAP fusion area -0.0102, pedestrian AP -0.017 (feature shift of a network trained with L4, not proof for a retrain). "
                     "Result in the D65 row. Was: " + r["note"])
        n += 1
assert n == 1
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
w.writeheader(); w.writerows(rows)
open(path, "w", newline="", encoding="utf-8").write(buf.getvalue())
print("schedule note changed", n)
PYEOF
python3 ../autoresearch/tools/schedule.py show 2>&1 | grep -E "L4|done \(" | cut -c1-200

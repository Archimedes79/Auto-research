#!/bin/bash
# D7 step 4: GaussianCaR features of the 255 new scenes (b456), three runs through the launcher (ledger row, commit + tag lcr/<id>, done flag only on success).
# At most two GPU processes at once: train halves in parallel, the val part (CONFIRM) after the first half. Usage: wsl bash training/data/prep/run_b456_extract.sh (moved from temp/loop19/ on 2026-10-31, D24)
cd "$(dirname "$0")/../../.." || exit 1                   # the problem folder
RUN="python3 ../autoresearch/tools/run.py"
X="bash training/data/prep/run_gcar.sh gcar_tools.py extract --split b456 --workers 2"
HYP="data extension b456 (D7): GaussianCaR features of the new frames with the b123 PCA basis"
PRED="every row of the part written, finite, value range like b123 (dec32 std about 1.5, logit7 std about 4)"
RULE="b456 is used for TRAIN2 / CONFIRM / UNSEEN only if the post-check passes (row count, finite, std within 20 % of b123)"
nohup sleep 14400 >/dev/null 2>&1 &                       # keeps the WSL alive while the jobs run
$RUN --id E234.tr0 --hyp "$HYP" --pred "$PRED" --rule "$RULE" --basis E27 --retries 1 -- $X --part train --start 0 --end 4366 > /tmp/E234.tr0.out 2>&1 &
p0=$!
sleep 20
$RUN --id E234.tr1 --hyp "$HYP" --pred "$PRED" --rule "$RULE" --basis E27 --retries 1 -- $X --part train --start 4366 --end 8731 > /tmp/E234.tr1.out 2>&1 &
p1=$!
sleep 20
$RUN --id E234.val --hyp "$HYP" --pred "$PRED" --rule "$RULE" --basis E27 --retries 1 --after E234.tr0 -- $X --part val > /tmp/E234.val.out 2>&1 &
p2=$!
wait $p0 $p1 $p2                                          # not the keep-alive sleep
tail -n 2 /tmp/E234.tr0.out /tmp/E234.tr1.out /tmp/E234.val.out

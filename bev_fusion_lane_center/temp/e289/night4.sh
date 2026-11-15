#!/bin/bash
# E289 step 1b on the GPU: the LINE stability probe of B (shift_probe2.py), both seeds, through the launcher without a card (a diagnostic: it feeds no verdict). Waits for the E287 runs (temp/e287/LANES_DONE).
cd "$(dirname "$0")/../.."
L=temp/e289/night4.log
echo "waiting for temp/e287/LANES_DONE $(date)" >> $L
until [ -f temp/e287/LANES_DONE ]; do sleep 60; done
H=$(date +%-H)
if ! { [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { [ "$H" -ge 8 ] && [ "$H" -lt 17 ]; }; }; then echo "outside the GPU window $(date), stopping" >> $L; exit 0; fi
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E289.probe2_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh shift_probe2.py $S >> temp/e289/probe2_s$S.log 2>&1
  echo "probe2 s$S exit $? $(date)" >> $L
done
touch temp/e289/PROBE2_DONE
echo "done $(date)" >> $L

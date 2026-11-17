#!/bin/bash
# E348 stage 0: the unsnapped and the attach-only cards of the CONTROL E324 s, scored from the stored dumps (CPU, by hand: replay); new files, nothing is overwritten
cd "$(dirname "$0")/../.."
for S in 0 1; do
  ( bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --attach --thr 0.1 > temp/e348/stage0_attach_s$S.log 2>&1
    bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --thr 0.1 > temp/e348/stage0_unsnap_s$S.log 2>&1 ) &
done
wait
echo "stage 0 done $(date)" > temp/e348/stage0.done

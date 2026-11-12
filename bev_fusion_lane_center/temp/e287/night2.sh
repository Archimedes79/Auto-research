#!/bin/bash
# E287 chain: waits for the E286 scores (temp/e286/SCORED), takes the best junction-maps arm, smoke, then two lanes (seed 0 / seed 1) of p2, p1, p0, then dump + score of every run.
# Log temp/e287/night2.log; flags SMOKE_OK, LANES_DONE, SCORED. GPU only inside the window (the chain ends by itself when the window is over: a lane starts no run after 07:00).
cd "$(dirname "$0")/../.."
L=temp/e287/night2.log
echo "waiting for temp/e286/SCORED $(date)" >> $L
until [ -f temp/e286/SCORED ]; do sleep 60; done
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
echo "E286 best arm: $BEST $(date)" >> $L
[ -n "$BEST" ] || { echo "no best arm" >> $L; exit 1; }
in_window() { H=$(date +%-H); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { [ "$H" -ge 8 ] && [ "$H" -lt 17 ]; }; }
if ! in_window; then echo "outside the GPU window $(date), stopping" >> $L; exit 0; fi
if ! bash temp/e287/start_pairs.sh p2 0 $BEST smoke >> $L 2>&1; then echo "SMOKE FAILED $(date)" >> $L; exit 1; fi
touch temp/e287/SMOKE_OK
lane() {
  S=$1
  for ARM in p2 p1 p0; do
    if ! in_window; then echo "lane $S: window over before $ARM $(date)" >> $L; return; fi
    bash temp/e287/start_pairs.sh $ARM $S $BEST >> temp/e287/lane$S.log 2>&1
    echo "lane $S $ARM exit $? $(date)" >> $L
  done
}
lane 0 &
lane 1 &
wait
touch temp/e287/LANES_DONE
for S in 0 1; do
  for ARM in p2 p1 p0; do
    if [ -f runs/e287_${ARM}_s$S/model.pt ]; then
      bash tools/env/_pyr.sh jends_eval.py dump --run e287_${ARM}_s$S >> temp/e287/dump_${ARM}_s$S.log 2>&1
      bash tools/env/_py14.sh jends_eval.py score --run e287_${ARM}_s$S --seed $S >> temp/e287/score_${ARM}_s$S.log 2>&1
      echo "scored $ARM $S exit $? $(date)" >> $L
    fi
  done
done
touch temp/e287/SCORED
echo "done $(date)" >> $L

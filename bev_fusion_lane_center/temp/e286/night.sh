#!/bin/bash
# E286 night chain: waits for the GPU window (22:00), smoke of one arm, then two lanes (seed 0 / seed 1) of j1, j0, j2, then the scores. Log: temp/e286/night.log; flags: SMOKE_OK, LANES_DONE, SCORED.
cd "$(dirname "$0")/../.."
L=temp/e286/night.log
echo "waiting for 22:00 $(date)" >> $L
until [ "$(date +%-H)" -ge 22 ] || [ "$(date +%-H)" -lt 7 ]; do sleep 30; done
echo "window open $(date)" >> $L
if ! bash temp/e286/start_jmaps.sh j1 0 smoke >> $L 2>&1; then echo "SMOKE FAILED $(date)" >> $L; exit 1; fi
touch temp/e286/SMOKE_OK
lane() {
  S=$1
  for ARM in j1 j0 j2; do
    bash temp/e286/start_jmaps.sh $ARM $S >> temp/e286/lane$S.log 2>&1
    echo "lane $S $ARM exit $? $(date)" >> $L
  done
}
lane 0 &
lane 1 &
wait
touch temp/e286/LANES_DONE
for S in 0 1; do
  for ARM in j1 j0 j2; do
    if [ -f runs/e286_${ARM}_s$S/model.pt ]; then
      bash tools/env/_pyr.sh jmaps_eval.py score --run e286_${ARM}_s$S >> temp/e286/score_${ARM}_s$S.log 2>&1
      echo "score $ARM $S exit $? $(date)" >> $L
    fi
  done
done
touch temp/e286/SCORED
echo "done $(date)" >> $L

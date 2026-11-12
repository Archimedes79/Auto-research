#!/bin/bash
# Day chain 7 (2026-11-12 12:15): E313 (end heat sigma 0.7) on both seeds as soon as the two E312 pair trainings have ended (GPU slots); scored when done. Starts only inside the GPU window (weekdays 08:30 - 16:30 for a START). Log temp/day7/day7.log; flag DAY7_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day7/day7.log
mkdir -p temp/day7
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 60; done; }
echo "day7 waits for the E312 trainings $(date)" >> $L
sleep 120
until [ -z "$(pgrep -f '[j]ends_train.py --name e312_p')" ]; do sleep 15; done
echo "E312 trainings ended $(date)" >> $L
lane() {
  S=$1
  wait_start; bash temp/e313/start_sigma.sh $S >> temp/day7/lane$S.log 2>&1; echo "lane $S E313 exit $? $(date)" >> $L
  bash tools/env/_pyr.sh jmaps_eval.py score --run e313_s_s$S >> temp/day7/score_e313_s$S.log 2>&1
  echo "scored e313_s_s$S $(date)" >> $L
}
lane 0 &
lane 1 &
wait
touch temp/day7/DAY7_DONE
echo "done $(date)" >> $L

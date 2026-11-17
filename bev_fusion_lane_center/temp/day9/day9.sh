#!/bin/bash
# Day chain 9 (2026-11-12 14:15): E314 (the E312 pair head on the sigma 0.7 maps of E313), both seeds, after the E298 trainings (GPU slots); lean post-processing (dump with flock, snapped p >= 0.1 first, then the scorer of record, snapped p >= 0.5, PR against E312, maneuver). Starts only inside the GPU window (weekdays 08:30 - 16:30 for a START). Log temp/day9/day9.log; flag DAY9_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day9/day9.log
mkdir -p temp/day9
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 30; done; }
echo "day9 waits for the E298 trainings $(date)" >> $L
until [ -z "$(pgrep -f '[j]ends_train.py --name e298_w')" ]; do sleep 15; done
echo "E298 trainings ended $(date)" >> $L
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day9/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day9/score_snap_$R.log 2>&1
  echo "snap scored $R $(date)" >> $L
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day9/score_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.5 >> temp/day9/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e312_p_s$S >> temp/day9/pr_$R.log 2>&1
  echo "scored $R $(date)" >> $L
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1s >> temp/day9/maneuver_$R.log 2>&1
  echo "maneuver $R $(date)" >> $L
}
lane() {
  S=$1
  wait_start; bash temp/e314/start_pairs.sh $S >> temp/day9/lane$S.log 2>&1; echo "lane $S E314 exit $? $(date)" >> $L
  post_pair e314_p_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day9/DAY9_DONE
echo "done $(date)" >> $L

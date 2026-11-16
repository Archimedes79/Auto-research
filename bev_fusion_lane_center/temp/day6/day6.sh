#!/bin/bash
# Day chain 6 (2026-11-12 12:00, replaces day5 after the E307 result): card E312, per lane (seed 0 / seed 1): the E305-recipe pair head with the E303 handle scales on the E307 maps (sigma 1.0), then dump, scorer of record, snapped read-outs, PR against E305, maneuver coverage, score of the jointly trained maps.
# Dumps are serialised with flock (host RAM: two dumps beside trainings die with 'Cannot allocate memory'). Starts only inside the GPU window (nights 22 - 07, weekdays 08:30 - 16:30 for a START). Log temp/day6/day6.log; flag DAY6_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day6/day6.log
mkdir -p temp/day6
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 60; done; }
until [ -z "$(pgrep -f '[j]maps_train.py --name e304')" ]; do sleep 5; done
echo "day6 start $(date)" >> $L
post_maps() {
  R=$1
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jmaps_eval.py score --run $R >> temp/day6/score_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day6/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day6/score_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day6/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.5 >> temp/day6/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --chord --thr 0.1 >> temp/day6/score_snap_chord_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e305_e_s$S >> temp/day6/pr_$R.log 2>&1
  echo "scored $R $(date)" >> $L
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.5s p0.1s p0.1 >> temp/day6/maneuver_$R.log 2>&1
  echo "maneuver $R $(date)" >> $L
  post_maps ${R}_maps
}
lane() {
  S=$1
  wait_start; bash temp/e312/start_pairs.sh $S >> temp/day6/lane$S.log 2>&1; echo "lane $S pairs exit $? $(date)" >> $L
  post_pair e312_p_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day6/DAY6_DONE
echo "done $(date)" >> $L

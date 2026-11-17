#!/bin/bash
# Day chain 5 (2026-11-12): after day chain 4 (E303 / E307 / E304): the combined end maps m2 (heat sigma 1.0 + far-range weight 2) and the E305 pair head on them (card E312), per lane (seed 0 / seed 1); each run scored in the background (maps score, dump, scorer of record, snapped read-out at p >= 0.1 / 0.5,
# pair PR against E305, maneuver coverage). Starts only inside the GPU window (nights 22 - 07, weekdays 08:30 - 16:30 for a START). Log temp/day5/day5.log; flag DAY5_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day5/day5.log
mkdir -p temp/day5
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 60; done; }
echo "day5 waits for DAY4_DONE $(date)" >> $L
until [ -f temp/day4/DAY4_DONE ]; do sleep 30; done
echo "day4 done $(date)" >> $L
post_maps() {
  R=$1
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jmaps_eval.py score --run $R >> temp/day5/score_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day5/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day5/score_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day5/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.5 >> temp/day5/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e305_e_s$S >> temp/day5/pr_$R.log 2>&1
  echo "scored $R $(date)" >> $L
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.5s p0.1s p0.1 >> temp/day5/maneuver_$R.log 2>&1
  echo "maneuver $R $(date)" >> $L
  post_maps ${R}_maps
}
lane() {
  S=$1
  wait_start; bash temp/e312/start_maps.sh $S >> temp/day5/lane$S.log 2>&1; echo "lane $S maps exit $? $(date)" >> $L
  post_maps e312_m2_s$S &
  wait_start; bash temp/e312/start_pairs.sh $S >> temp/day5/lane$S.log 2>&1; echo "lane $S pairs exit $? $(date)" >> $L
  post_pair e312_p_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day5/DAY5_DONE
echo "done $(date)" >> $L

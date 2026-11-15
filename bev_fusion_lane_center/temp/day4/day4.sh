#!/bin/bash
# Day chain 4 (replaces day3.sh from the end of E305 on). The maneuver analysis of 2026-11-12 (temp/jr/maneuver.py, E308 dumps) says: a perfect pair classifier on today's keypoints realises 59 % of the GT maneuvers (the plain graph 50 %), and of the maneuvers
# whose two ends are found only 67 % even with perfect pairs: the connector SHAPE (chord-tangent Hermite curve) and the ends that are not found are worth more than the pair decision. So per lane (seed 0 / seed 1):
# E303 (handle scales = the connector shape) -> E307 (narrower end heat) -> E304 (far-range maps) -> E306 (exact raster) -> E298 (driven weight) -> E302 (capacitated pairs).
# E305 (maps + pair head together) was started by day3.sh and is still training: wait for it. Every run is dumped / scored in the background beside the next training (+ the maneuver coverage for pair runs).
# Starts only inside the GPU window (nights 22 - 07, weekdays 08:30 - 16:30 for a START). Log temp/day4/day4.log; flag temp/day4/DAY4_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day4/day4.log
mkdir -p temp/day4
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 60; done; }
BEST=j1
CTX=dil
echo "day4 start $(date): waiting for the E305 runs of day3" >> $L
until [ -z "$(pgrep -f '[j]ends_train.py --name e305_e_s')" ]; do sleep 20; done
echo "E305 trainings ended $(date)" >> $L
post_pair() {   # dump + score + PR + maneuver coverage of a pair-head run
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day4/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day4/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e287_p2_s$S >> temp/day4/pr_$R.log 2>&1
  echo "scored $R $(date)" >> $L
  bash tools/env/_py14.sh ../jr/maneuver.py $R $S plain 'p>=0.5' 'p>=0.1' >> temp/day4/maneuver_$R.log 2>&1
  echo "maneuver $R $(date)" >> $L
}
post_maps() {   # score of a maps-head run (extra args: --raster exact for a control)
  R=$1; shift
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jmaps_eval.py score --run $R "$@" >> temp/day4/score_$R.log 2>&1
  echo "scored $R $* $(date)" >> $L
}
lane() {
  S=$1
  ( post_pair e305_e_s$S $S; post_maps e305_e_s${S}_maps ) &
  wait_start; bash temp/e303/start_scales.sh $S $BEST >> temp/day4/lane$S.log 2>&1; echo "lane $S E303 exit $? $(date)" >> $L
  ( post_pair e303_s_s$S $S
    if [ -f runs/e303_s_s$S/model.pt ]; then
      mkdir -p runs/e303_chord_s$S && cp runs/e303_s_s$S/ends_val.pkl runs/e303_s_s$S/model.pt runs/e303_chord_s$S/ 2>/dev/null
      bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s$S --seed $S --chord >> temp/day4/score_e303_chord_s$S.log 2>&1
    fi ) &
  wait_start; bash temp/e307/start_narrow.sh $S >> temp/day4/lane$S.log 2>&1; echo "lane $S E307 exit $? $(date)" >> $L
  post_maps e307_n_s$S &
  wait_start; bash temp/e304/start_far.sh $S $CTX >> temp/day4/lane$S.log 2>&1; echo "lane $S E304 exit $? $(date)" >> $L
  post_maps e304_f_s$S &
  wait_start; bash temp/e306/start_exact.sh $S >> temp/day4/lane$S.log 2>&1; echo "lane $S E306 exit $? $(date)" >> $L
  ( post_maps e306_x_s$S; post_maps e286_j1_s$S --raster exact ) &
  wait_start; bash temp/e298/start_driven.sh $S $BEST >> temp/day4/lane$S.log 2>&1; echo "lane $S E298 exit $? $(date)" >> $L
  post_pair e298_w_s$S $S &
  wait_start; bash temp/e302/start_cap.sh $S $BEST >> temp/day4/lane$S.log 2>&1; echo "lane $S E302 exit $? $(date)" >> $L
  post_pair e302_k_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day4/DAY4_DONE
echo "done $(date)" >> $L

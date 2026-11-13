#!/bin/bash
# E319 chain (NOT started; the parent decides when): both seeds in the two GPU slots, per lane: maps with dirx -> maps score -> kill check -> E312 pair recipe on them -> dump, snapped score, plain score, pair PR against E312, maneuver coverage.
# Starts only inside the GPU window (nights 22-07, Mon-Fri 08:30-16:30 for a START). usage: chain.sh [marker file to wait for, e.g. temp/day11/DAY11_DONE]. Log temp/e319/chain.log. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/e319/chain.log
mkdir -p temp/e319
[ -n "$1" ] && { echo "chain waits for $1 $(date)" >> $L; until [ -f "$1" ]; do sleep 30; done; echo "marker found $(date)" >> $L; }
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 30; done; }
post_pair() {
  R=$1; S=$2; C=$3
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/e319/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/e319/score_snap_$R.log 2>&1
  echo "snap scored $R $(date)" >> $L
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/e319/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control $C >> temp/e319/pr_$R.log 2>&1
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1s >> temp/e319/maneuver_$R.log 2>&1
  bash tools/env/_pyr.sh jmaps_eval.py score --run ${R}_maps >> temp/e319/score_${R}_maps.log 2>&1
  echo "scored $R $(date)" >> $L
}
lane() {
  S=$1
  wait_start; bash temp/e319/start_maps.sh $S >> temp/e319/lane$S.log 2>&1; echo "lane $S E319 maps exit $? $(date)" >> $L
  bash tools/env/_pyr.sh jmaps_eval.py score --run e319_m_s$S >> temp/e319/score_e319_m_s$S.log 2>&1
  if (cd temp/e319 && bash ../../tools/env/_py14.sh e319/kill_check.py $S >> kill_s$S.log 2>&1); then
    wait_start; bash temp/e319/start_pairs.sh $S >> temp/e319/lane$S.log 2>&1; echo "lane $S E319 pairs exit $? $(date)" >> $L
    post_pair e319_p_s$S $S e312_p_s$S
  else
    echo "E319 kill criterion met for seed $S: pair stage not run $(date)" >> $L
  fi
}
lane 0 &
lane 1 &
wait
touch temp/e319/E319_DONE
echo "done $(date)" >> $L

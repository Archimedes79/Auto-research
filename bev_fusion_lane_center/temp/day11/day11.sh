#!/bin/bash
# Day chain 11 (2026-11-12 21:40): E315 (maps 6000 steps, then the E312 pair head on them) and E316 (pair head 6000 steps on the E307 maps), both seeds in the two GPU slots, in the GPU window only (nights 22-07, Mon-Fri 08:30-17:30); dump with flock, snapped p >= 0.1 first.
cd "$(dirname "$0")/../.."
L=temp/day11/day11.log
mkdir -p temp/day11
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 30; done; }
post_pair() {
  R=$1; S=$2; C=$3
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day11/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day11/score_snap_$R.log 2>&1
  echo "snap scored $R $(date)" >> $L
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day11/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control $C >> temp/day11/pr_$R.log 2>&1
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1s >> temp/day11/maneuver_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
lane() {
  S=$1
  wait_start; bash temp/e315/start_maps.sh $S >> temp/day11/lane$S.log 2>&1; echo "lane $S E315 maps exit $? $(date)" >> $L
  bash tools/env/_pyr.sh jmaps_eval.py score --run e315_m_s$S >> temp/day11/score_e315_m_s$S.log 2>&1
  if (cd temp/e315 && bash ../../tools/env/_py14.sh e315/kill_check.py $S >> ../day11/kill_s$S.log 2>&1); then
    wait_start; bash temp/e315/start_pairs.sh $S >> temp/day11/lane$S.log 2>&1; echo "lane $S E315 pairs exit $? $(date)" >> $L
    post_pair e315_p_s$S $S e312_p_s$S &
  else
    echo "E315 kill criterion met for seed $S: pair stage not run $(date)" >> $L
  fi
  wait_start; bash temp/e316/start_pairs.sh $S >> temp/day11/lane$S.log 2>&1; echo "lane $S E316 exit $? $(date)" >> $L
  post_pair e316_p_s$S $S e312_p_s$S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day11/DAY11_DONE
echo "done $(date)" >> $L

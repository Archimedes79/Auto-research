#!/bin/bash
# Day chain 8 (2026-11-12 14:03): E298 on the E312 system (driven-pair weight 2; amendment of the card), both seeds, started while E313 is being scored so that the GPU does not idle; if E313 wins it is stopped by hand and the slots go to the pair head on the sigma 0.7 maps.
# Dumps serialised with flock. Starts only inside the GPU window (weekdays 08:30 - 16:30 for a START). Log temp/day8/day8.log; flag DAY8_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day8/day8.log
mkdir -p temp/day8
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day8/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day8/score_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day8/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e312_p_s$S >> temp/day8/pr_$R.log 2>&1
  bash tools/env/_py14.sh jsplits.py score --run e312_p_s$S --run $R >> temp/day8/splits_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
lane() {
  S=$1
  bash temp/e298/start_driven_e312.sh $S >> temp/day8/lane$S.log 2>&1; echo "lane $S E298 exit $? $(date)" >> $L
  post_pair e298_w_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day8/DAY8_DONE
echo "done $(date)" >> $L

#!/bin/bash
# Day chain 10 (2026-11-12 14:40): E298 on the E312 system (driven-pair weight 2; amendment of the card), both seeds, after the E314 trainings (GPU slots); dump with flock, snapped p >= 0.1 first, record scorer, PR against E312, driven split (jsplits). Starts only inside the GPU window (weekdays 08:30 - 16:30 for a START). Log temp/day10/day10.log; flag DAY10_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day10/day10.log
mkdir -p temp/day10
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
echo "day10 waits for the E314 trainings $(date)" >> $L
sleep 240
until [ -z "$(pgrep -f '[j]ends_train.py --name e314_p')" ]; do sleep 15; done
echo "E314 trainings ended $(date)" >> $L
can_start || { echo "outside the GPU window, E298 not started $(date)" >> $L; exit 0; }
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day10/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day10/score_snap_$R.log 2>&1
  echo "snap scored $R $(date)" >> $L
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day10/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e312_p_s$S >> temp/day10/pr_$R.log 2>&1
  bash tools/env/_py14.sh jsplits.py score --run e312_p_s$S --run $R >> temp/day10/splits_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
lane() {
  S=$1
  bash temp/e298/start_driven_e312.sh $S >> temp/day10/lane$S.log 2>&1; echo "lane $S E298 exit $? $(date)" >> $L
  post_pair e298_w_s$S $S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day10/DAY10_DONE
echo "done $(date)" >> $L

#!/bin/bash
# Day chain 12b (2026-11-12 23:20, lane lead), replaces day12 under D85 (Mirko 2026-11-12 23:05: E320, the differentiable attachment instead of the snap, has priority; E317 / E318 / E319 wait until E320 is measured and carry hold markers; E315 and E316 run on in day11).
# After day11: (1) the idle-GPU timing of the junction heads, (2) E320 as soon as temp/e320/READY exists (the first line of READY = the run-name prefix), both seeds in the two GPU slots, followed by dump, snapped / unsnapped / plain scores, pair PR against E312, maneuver coverage.
# Starts only in the GPU windows (nights 22:00-05:00, Mon-Fri 08:30-15:30). Nothing else is started here; the verdict on E320 and the clearing of the holds are Mirko's. Never edit this script while it runs. Log temp/day12/day12b.log.
cd "$(dirname "$0")/../.."
L=temp/day12/day12b.log
mkdir -p temp/day12
echo "day12b waits for DAY11_DONE $(date)" >> $L
until [ -f temp/day11/DAY11_DONE ]; do sleep 30; done
echo "day11 done $(date)" >> $L
can_start() { H=$(date +%-H); M=$(date +%-M); DOW=$(date +%u); [ "$H" -ge 22 ] || [ "$H" -lt 5 ] || { [ "$DOW" -le 5 ] && { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 15 ] || { [ "$H" -eq 15 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 30; done; }
post_pair() {
  R=$1; S=$2; C=$3
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day12/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> temp/day12/score_snap_$R.log 2>&1
  echo "snap scored $R $(date)" >> $L
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --thr 0.1 >> temp/day12/score_unsnap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day12/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control $C >> temp/day12/pr_$R.log 2>&1
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1 p0.1s >> temp/day12/maneuver_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
pair_arm() {
  E=$1; LOW=$(echo $E | tr 'E' 'e'); PFX=$(head -1 temp/$LOW/READY | tr -d '[:space:]')
  for S in 0 1; do
    ( wait_start; bash temp/$LOW/start_pairs.sh $S >> temp/day12/${LOW}_lane$S.log 2>&1; echo "$E seed $S exit $? $(date)" >> $L; post_pair ${PFX}$S $S e312_p_s$S ) &
  done
  wait
  touch temp/day12/${E}_DONE
  echo "$E done $(date)" >> $L
}
# (1) timing on an idle GPU
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-9999}" -lt 2500 ]; then
  bash tools/env/_py14.sh jr/timing_junction.py e312_p_s0 --n 60 >> temp/day12/timing.log 2>&1; echo "timing done $(date)" >> $L
else
  echo "timing skipped (GPU in use: $USED MiB) $(date)" >> $L
fi
# (2) E320 first (D85): wait for its READY, then both seeds
until [ -f temp/e320/READY ]; do sleep 60; done
echo "E320 READY $(date)" >> $L
pair_arm E320
touch temp/day12/DAY12B_DONE
echo "day12b done $(date)" >> $L

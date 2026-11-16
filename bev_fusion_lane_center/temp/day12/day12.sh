#!/bin/bash
# Day chain 12 (2026-11-12 23:00, lane lead): after day11 (E315 / E316) the new differentiable experiments, in the GPU windows only (nights 22-07 for a start before 05:00, Mon-Fri 08:30-15:30 for a start).
# Order: (1) the idle-GPU timing of the junction heads, (2) the first READY experiment of E317 (geometry loss) else E319 (heading-guided exchange), (3) the others as they become READY (temp/e<N>/READY written by the builder).
# Experiments run as pair arms (both seeds in the two GPU slots), each followed by dump, snapped score, plain score, pair PR against E312, maneuver coverage. Never edit this script while it runs; log temp/day12/day12.log.
cd "$(dirname "$0")/../.."
L=temp/day12/day12.log
mkdir -p temp/day12
echo "day12 waits for DAY11_DONE $(date)" >> $L
until [ -f temp/day11/DAY11_DONE ]; do sleep 30; done
echo "day11 done $(date)" >> $L
# a start is allowed 22:00-05:00 (a 2 h job still ends before 07:00) and Mon-Fri 08:30-15:30 (ends before 17:30)
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
# one pair-stage experiment, both seeds in parallel: pair_arm <exp id e.g. E317>; the run name prefix (e.g. e317_g_s) is the first line of temp/e317/READY
pair_arm() {
  E=$1; LOW=$(echo $E | tr 'E' 'e'); PFX=$(head -1 temp/$LOW/READY | tr -d '[:space:]')
  for S in 0 1; do
    ( wait_start; bash temp/$LOW/start_pairs.sh $S >> temp/day12/${LOW}_lane$S.log 2>&1; echo "$E seed $S exit $? $(date)" >> $L; post_pair ${PFX}$S $S e312_p_s$S ) &
  done
  wait
  touch temp/day12/${E}_DONE
  echo "$E done $(date)" >> $L
}
# (1) timing on an idle GPU (nothing else may train: gate on memory in use)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-9999}" -lt 2500 ]; then
  bash tools/env/_py14.sh jr/timing_junction.py e312_p_s0 --n 60 >> temp/day12/timing.log 2>&1; echo "timing done $(date)" >> $L
else
  echo "timing skipped (GPU in use: $USED MiB) $(date)" >> $L
fi
# (2) + (3) the experiments, in this order, each only when it is READY at its turn (E319 has its own chain script)
if [ -f temp/e317/READY ]; then pair_arm E317; fi
if [ -f temp/e319/chain.sh ]; then
  can_start || wait_start
  bash temp/e319/chain.sh >> temp/day12/e319_chain.out 2>&1
fi
if [ -f temp/e317/READY ] && [ ! -f temp/day12/E317_DONE ]; then pair_arm E317; fi
if [ -f temp/e318/READY ]; then pair_arm E318; fi
if [ -f temp/e320/READY ]; then pair_arm E320; fi
touch temp/day12/DAY12_DONE
echo "day12 done $(date)" >> $L

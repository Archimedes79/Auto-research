#!/bin/bash
# Day chain 1 (after the night chains): E304 (far-range maps), E298 (driven weight), E303 (handle scales), two lanes (seed 0 / seed 1), each lane: E304 -> E298 -> E303, then the scores.
# Waits for temp/e287/JJ_DONE (the joint arm + q2) and for the GPU window (nights 22 - 07, 08:30 - 17:30). Log temp/day1/day.log; flags DAY1_DONE.
cd "$(dirname "$0")/../.."
L=temp/day1/day.log
in_window() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 17 ] || { [ "$H" -eq 17 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_window() { until in_window; do sleep 60; done; }
echo "waiting for temp/e287/JJ_DONE $(date)" >> $L
until [ -f temp/e287/JJ_DONE ]; do sleep 60; done
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
CTX=$(case $BEST in j0) echo none;; j1) echo dil;; j2) echo attn;; esac)
echo "E286 best arm: $BEST ($CTX) $(date)" >> $L
lane() {
  S=$1
  wait_window; bash temp/e304/start_far.sh $S $CTX >> temp/day1/lane$S.log 2>&1; echo "lane $S E304 exit $? $(date)" >> $L
  wait_window; bash temp/e298/start_driven.sh $S $BEST >> temp/day1/lane$S.log 2>&1; echo "lane $S E298 exit $? $(date)" >> $L
  wait_window; bash temp/e303/start_scales.sh $S $BEST >> temp/day1/lane$S.log 2>&1; echo "lane $S E303 exit $? $(date)" >> $L
}
lane 0 &
lane 1 &
wait
for S in 0 1; do
  [ -f runs/e304_f_s$S/model.pt ] && bash tools/env/_pyr.sh jmaps_eval.py score --run e304_f_s$S >> temp/day1/score_e304_s$S.log 2>&1
  for R in e298_w_s$S e303_s_s$S; do
    if [ -f runs/$R/model.pt ]; then
      bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day1/dump_$R.log 2>&1
      bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day1/score_$R.log 2>&1
    fi
  done
  if [ -f runs/e303_s_s$S/model.pt ]; then
    mkdir -p runs/e303_chord_s$S && cp runs/e303_s_s$S/ends_val.pkl runs/e303_s_s$S/model.pt runs/e303_chord_s$S/ 2>/dev/null
    bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s$S --seed $S --chord >> temp/day1/score_e303_chord_s$S.log 2>&1
  fi
  echo "scored seed $S $(date)" >> $L
done
touch temp/day1/DAY1_DONE
echo "done $(date)" >> $L

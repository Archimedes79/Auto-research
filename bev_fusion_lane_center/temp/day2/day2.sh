#!/bin/bash
# Day chain 2 (after day chain 1): E306 (exact polygon targets) and E305 (maps + pair head trained together), two lanes (seed 0 / seed 1), then the scores. Waits for temp/day1/DAY1_DONE and the GPU window.
# Log temp/day2/day2.log; flags DAY2_DONE.
cd "$(dirname "$0")/../.."
L=temp/day2/day2.log
in_window() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 17 ] || { [ "$H" -eq 17 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_window() { until in_window; do sleep 60; done; }
echo "waiting for temp/day1/DAY1_DONE $(date)" >> $L
until [ -f temp/day1/DAY1_DONE ]; do sleep 60; done
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
echo "E286 best arm: $BEST $(date)" >> $L
lane() {
  S=$1
  wait_window; bash temp/e306/start_exact.sh $S >> temp/day2/lane$S.log 2>&1; echo "lane $S E306 exit $? $(date)" >> $L
  wait_window; bash temp/e305/start_e2e.sh e $S $BEST >> temp/day2/lane$S.log 2>&1; echo "lane $S E305 exit $? $(date)" >> $L
}
lane 0 &
lane 1 &
wait
for S in 0 1; do
  if [ -f runs/e306_x_s$S/model.pt ]; then
    bash tools/env/_pyr.sh jmaps_eval.py score --run e306_x_s$S >> temp/day2/score_e306_s$S.log 2>&1
    bash tools/env/_pyr.sh jmaps_eval.py score --run e286_j1_s$S --raster exact >> temp/day2/score_e306_ctl_s$S.log 2>&1
  fi
  R=e305_e_s$S
  if [ -f runs/$R/model.pt ]; then
    bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day2/dump_$R.log 2>&1
    bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day2/score_$R.log 2>&1
    bash tools/env/_pyr.sh jmaps_eval.py score --run ${R}_maps >> temp/day2/score_${R}_maps.log 2>&1
  fi
  echo "scored seed $S $(date)" >> $L
done
touch temp/day2/DAY2_DONE
echo "done $(date)" >> $L

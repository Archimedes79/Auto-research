#!/bin/bash
# Day chain 3 (after day chain 2): E302 (capacitated Sinkhorn pair head), two lanes (seed 0 / seed 1), then the scores (jpair_pr against E287 p2, jends_eval). Waits for temp/day2/DAY2_DONE and the GPU window. Log temp/day3/day3.log; flags DAY3_DONE.
cd "$(dirname "$0")/../.."
L=temp/day3/day3.log
in_window() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 17 ] || { [ "$H" -eq 17 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_window() { until in_window; do sleep 60; done; }
echo "waiting for temp/day2/DAY2_DONE $(date)" >> $L
until [ -f temp/day2/DAY2_DONE ]; do sleep 60; done
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
echo "E286 best arm: $BEST $(date)" >> $L
lane() { S=$1; wait_window; bash temp/e302/start_cap.sh $S $BEST >> temp/day3/lane$S.log 2>&1; echo "lane $S E302 exit $? $(date)" >> $L; }
lane 0 &
lane 1 &
wait
for S in 0 1; do
  R=e302_k_s$S
  if [ -f runs/$R/model.pt ]; then
    bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day3/dump_$R.log 2>&1
    bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day3/score_$R.log 2>&1
    bash tools/env/_py14.sh jpair_pr.py score --run $R --control e287_p2_s$S >> temp/day3/pr_$R.log 2>&1
  fi
  echo "scored seed $S $(date)" >> $L
done
touch temp/day3/DAY3_DONE
echo "done $(date)" >> $L

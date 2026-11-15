#!/bin/bash
# Day chain 16 (2026-11-14, lane chat, order of the lane lead): E325 = the learned wiring head, arm w, both seeds (the launcher has two GPU slots). It starts nothing before temp/e325/READY and temp/e325/ACCEPTED (the code critic
# pass) exist, and only in the GPU windows (starts 22:00 - 05:00, Mon - Fri 08:30 - 15:30). Per seed: start_w.sh (E325.w_s<S>: the TRAIN records, the pilot, the heads), post_w.sh (the snapped read-out with the head, bootstraps), then
# the verdict script. Never edit this script while it runs.  Log temp/day16/day16.log, marker temp/day16/DAY16_DONE.
cd "$(dirname "$0")/../.."
L=temp/day16/day16.log; LOGD=temp/day16
mkdir -p $LOGD
echo "day16 starts $(date)" >> $L
. temp/e330/chain_lib.sh          # can_start, wait_start
until [ -f temp/e325/READY ] && [ -f temp/e325/ACCEPTED ]; do sleep 60; done
echo "e325 READY and ACCEPTED $(date)" >> $L
for S in 0 1; do
  ( wait_start; bash temp/e325/start_w.sh $S >> $LOGD/e325_w_lane$S.log 2>&1; echo "e325 w seed $S exit $? $(date)" >> $L; bash temp/e325/post_w.sh $S ) &
done
wait
bash tools/env/_py14.sh ../temp/e325/verdict_e325.py >> $LOGD/verdict.log 2>&1
echo "e325 verdict script exit $? $(date)" >> $L
touch $LOGD/DAY16_DONE
echo "day16 done $(date)" >> $L

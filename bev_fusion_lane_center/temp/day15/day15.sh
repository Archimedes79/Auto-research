#!/bin/bash
# Day chain 15 (2026-11-13, lane chat, order of the lane lead: E292 after E324): E292 = the fine-tune of B's final checkpoint with the pyramid frozen, arm c2 (no layer) for both seeds, then - when the PRECONDITION
# of the card holds (c2 topo F1 >= c - 0.01 on both seeds, temp/e292/pre_check.py) - arm a (the Matrix-NMS layer). It starts nothing before temp/e292/READY and temp/e292/ACCEPTED (the critic pass on the code) exist and
# the two E324 arm-s TRAINING runs are done (log lines of day14), and only in the GPU windows (starts 22:00 - 05:00, Mon - Fri 08:30 - 15:30). The launcher's two GPU slots serialise it with day14's later items.
# Never edit this script while it runs.  Log temp/day15/day15.log, marker temp/day15/DAY15_DONE.
cd "$(dirname "$0")/../.."
L=temp/day15/day15.log; LOGD=temp/day15
mkdir -p $LOGD
echo "day15 starts $(date)" >> $L
. temp/e330/chain_lib.sh          # can_start, wait_start
until [ -f temp/e292/READY ] && [ -f temp/e292/ACCEPTED ]; do sleep 60; done
echo "e292 READY and ACCEPTED $(date)" >> $L
until [ "$(grep -c 'e324 arm s seed . exit' temp/day14/day14.log 2>/dev/null)" -ge 2 ]; do sleep 60; done
echo "e324 arm s training done $(date)" >> $L
arm_pair() {   # <c2|a>: both seeds in parallel (the launcher has two GPU slots)
  for S in 0 1; do
    ( wait_start; bash temp/e292/start_arm.sh $1 $S >> $LOGD/e292_$1_lane$S.log 2>&1; echo "e292 $1 seed $S exit $? $(date)" >> $L; bash temp/e292/post_arm.sh $1 $S ) &
  done
  wait
}
arm_pair c2
python3 temp/e292/pre_check.py >> $LOGD/pre_check.log 2>&1; PRE=$?
echo "e292 precondition exit $PRE $(date)" >> $L
if [ $PRE = 0 ]; then arm_pair a; else echo "e292 arm a not run: the precondition (c2 against c) fails, the comparison would be void $(date)" >> $L; fi
touch $LOGD/DAY15_DONE
echo "day15 done $(date)" >> $L

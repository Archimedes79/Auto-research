#!/bin/bash
# Mirko's order of 2026-11-05: "Run them after spurmitte completes, then shutdown computer". Part 2 of the chain (the paint check ran at 18:44, see chain.log): the GPU part of the overfit test.
# Replaces the waiting second half of chain.sh. New gate (D57, lane first): the lane chat scores four models at 22:00 (temp/e275/launch_scoring.sh, flag SCORED in the lane folder); this chain starts after that flag,
# or after 01:30 if the flag never comes (the window ends 07:00, the GPU part needs about 100 minutes), and always needs a window with >= 100 minutes left and a free GPU.
#   1. diag.overfit8_before  the t20 baseline on 8 train scenes (all frames)
#   2. diag.overfit8_s0      train on these 8 scenes only, 600 steps, warm start from the baseline, no augmentation (pinned to the code of bf/t21.ctl_s0)
#   3. diag.overfit8_after   the same evaluation of the overfit network
# The shutdown is NOT in this script: I do it after reading the results and checking that nothing else runs.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
LOG=temp/lane_tests/chain.log
exec >> $LOG 2>&1
LANE_FLAG=../bev_fusion_lane_center/temp/e275/SCORED
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {                                   # minutes left in the current GPU window (0 = outside), as temp/e274/launch_window.sh of the lane chat
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|bev_suite|qgraph_|trainsuite" | grep -v $$ > /dev/null
}
lane_done_or_late() { [ -f $LANE_FLAG ] || { n=$(now_min); [ $n -ge 90 ] && [ $n -lt 420 ]; }; }
echo "chain2 waits for the lane scoring (flag $LANE_FLAG, or 01:30), a GPU window with >= 100 min and a free GPU: $(date)"
until lane_done_or_late && [ "$(window_left)" -ge 100 ] && gpu_free; do sleep 30; done
echo "start: window left $(window_left) min, lane flag $([ -f $LANE_FLAG ] && echo yes || echo no): $(date)"
python3 ../autoresearch/tools/run.py --id diag.overfit8_before --gpu --gpu-max-mb 3000 --split train \
  --hyp "Overfit test, reference: how well the finished baseline t20_base_s0 fits the lane layer on 8 of its own training scenes (all their frames)." \
  --pred "lane divider IoU@0.5 on the 8 scenes about 0.5 (train 0.54 over 25 train frames per scene in the earlier generalisation table; val 0.43)" \
  --rule "by hand: diagnostic, no verdict; the reference of diag.overfit8_after" \
  --retries 0 -- bash temp/lane_tests/eval8.sh runs/overfit8_before runs/t20_base_s0
echo "before exit $? $(date)"
# pinned to the code of the t21 control (tag bf/t21.ctl_s0): other chats change frozen files of training/ in the working tree (t22 stage B, default off); this diagnostic trains on the code its baseline used
python3 ../autoresearch/tools/run.py --id diag.overfit8_s0 --gpu --gpu-max-mb 3000 --split train --timeout 120 --pin bf/t21.ctl_s0 \
  --hyp "Overfit test: the network (loss, resolution, features) can fit the lane layer on a handful of scenes; the weak lane markings are then a matter of generalisation (data, supervision efficiency) or of label noise that averages out, not of capacity." \
  --pred "lane divider IoU@0.5 on the 8 training scenes rises from diag.overfit8_before (about 0.5) to >= 0.85 after 600 steps (6 clips x 3 keyframes per step, no augmentation, lr 1e-4)" \
  --rule "by hand: diagnostic, no verdict. After >= 0.85: the loss and the resolution can carry the labels (the gap is generalisation / data). After <= 0.65: even 8 scenes are not fitted: labels (noise, registration), loss or representation are the limit. Between: inconclusive, a longer run is the next step." \
  --retries 1 -- bash temp/train/run_ar.sh temp/compose/overfit8_s0.json
echo "overfit exit $? $(date)"
python3 ../autoresearch/tools/run.py --id diag.overfit8_after --gpu --gpu-max-mb 3000 --split train \
  --hyp "Overfit test, result: the lane layer of the overfit network on the same 8 training scenes." \
  --pred "lane divider IoU@0.5 >= 0.85" \
  --rule "by hand: diagnostic, no verdict; read with the rule of diag.overfit8_s0" \
  --retries 0 -- bash temp/lane_tests/eval8.sh runs/overfit8_s0 runs/overfit8_s0
echo "after exit $? $(date)"
touch temp/lane_tests/DONE
echo "chain done $(date)"

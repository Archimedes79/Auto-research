#!/bin/bash
# Mirko's order of 2026-11-05: "Run them after spurmitte completes, then shutdown computer". The two lane-marking tests, behind the lane project's work (D57: lane first):
#   gate 1: the flag temp/lane_tests/GO (I create it when the lane chat has said its queue is done), no lane / BEV job on the GPU
#   1. diag.paint_val        CPU: do the ground-truth lane lines lie on the paint? (temp/lane_tests/paint_vs_label.py)
#   gate 2: a GPU window (Mon-Fri 08:30-17:30 or nights 22:00-07:00) with >= 100 minutes left, a free GPU
#   2. diag.overfit8_before  the t20 baseline on 8 train scenes (all frames)
#   3. diag.overfit8_s0      train on these 8 scenes only, 600 steps, warm start from the baseline, no augmentation
#   4. diag.overfit8_after   the same evaluation of the overfit network
# The shutdown is NOT in this script: I do it after reading the results and checking that nothing else runs.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
LOG=temp/lane_tests/chain.log
exec >> $LOG 2>&1
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
echo "chain waits for temp/lane_tests/GO: $(date)"
until [ -f temp/lane_tests/GO ] && gpu_free; do sleep 30; done
echo "GO and a free GPU: $(date)"
PY=$HOME/miniconda3/envs/gcar/bin/python
python3 ../autoresearch/tools/run.py --id diag.paint_val --split val \
  --hyp "The stored ground-truth lane lines (lane_divider, road_divider) lie on the visible paint in the front camera image (within ~0.2 m at 5-20 m), so the weak lane-marking IoU (0.43 at 0.5 m) is not label noise. A 12-frame functional check of the script preceded this run (not a result; it hinted at weak agreement: hit rate 0.3-0.5 near, nothing beyond 20 m)." \
  --pred "lane_divider at 5-20 m: share of line points with a bright stripe within +-0.5 m of the label >= 0.6, the control (label moved 1.75 m sideways) <= 0.2; best offset per frame SD <= 0.25 m" \
  --rule "by hand: diagnostic, no verdict. Hit rate (+-0.5 m, 5-20 m, lane_divider) >= 0.6 and best-offset SD <= 0.25 m: the labels sit on the paint, look elsewhere (loss / representation). 0.3-0.6 or SD 0.25-0.5: the labels carry registration error of that size or lack paint, the thin-line target is partly unlearnable at this precision. <= 0.3 (near the control) or SD >= 0.5: the lines are not on the paint." \
  --retries 0 -- $PY temp/lane_tests/paint_vs_label.py --out temp/lane_tests/paint_out
echo "paint exit $? $(date)"
echo "chain waits for a GPU window with >= 100 min: $(date)"
until [ "$(window_left)" -ge 100 ] && gpu_free; do sleep 30; done
echo "window left $(window_left) min: $(date)"
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

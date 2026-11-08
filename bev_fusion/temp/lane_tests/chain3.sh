#!/bin/bash
# Retry of the overfit training and the evaluation after it (chain2.sh: --pin was refused for an id without an earlier attempt, so the training did not start and the evaluation after it failed on the missing model).
# Now WITHOUT --pin: run.py snapshots the working tree, which holds the t22 stage-B code of the Image Encoder chat (commit eebd9640, ModelConfig.camera_student, off by default; the config of this run does not set it). Written into the record as a deviation.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
LOG=temp/lane_tests/chain.log
exec >> $LOG 2>&1
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|bev_suite|qgraph_|trainsuite" | grep -v $$ > /dev/null
}
echo "chain3 waits for a free GPU: $(date)"
until gpu_free; do sleep 30; done
echo "chain3 start: $(date)"
python3 ../autoresearch/tools/run.py --id diag.overfit8_s0 --gpu --gpu-max-mb 3000 --split train --timeout 120 \
  --hyp "Overfit test: the network (loss, resolution, features) can fit the lane layer on a handful of scenes; the weak lane markings are then a matter of generalisation (data, supervision efficiency) or of label noise that averages out, not of capacity. Deviation from the plan: not pinned to the code of bf/t21.ctl_s0 (run.py refuses --pin for a first attempt); the working-tree code holds the t22 stage-B change (camera_student, off by default and not set in this config)." \
  --pred "lane divider IoU@0.5 on the 8 training scenes rises from diag.overfit8_before (0.463, measured) to >= 0.85 after 600 steps (6 clips x 3 keyframes per step, no augmentation, lr 1e-4)" \
  --rule "by hand: diagnostic, no verdict. After >= 0.85: the loss and the resolution can carry the labels (the gap is generalisation / data). After <= 0.65: even 8 scenes are not fitted: labels (noise, registration), loss or representation are the limit. Between: inconclusive, a longer run is the next step." \
  --retries 1 -- bash temp/train/run_ar.sh temp/compose/overfit8_s0.json
echo "overfit exit $? $(date)"
python3 ../autoresearch/tools/run.py --id diag.overfit8_after --gpu --gpu-max-mb 3000 --split train \
  --hyp "Overfit test, result: the lane layer of the overfit network on the same 8 training scenes." \
  --pred "lane divider IoU@0.5 >= 0.85" \
  --rule "by hand: diagnostic, no verdict; read with the rule of diag.overfit8_s0" \
  --retries 0 -- bash temp/lane_tests/eval8.sh runs/overfit8_s0 runs/overfit8_s0
echo "after exit $? $(date)"
touch temp/lane_tests/DONE3
echo "chain3 done $(date)"

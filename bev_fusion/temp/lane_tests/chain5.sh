#!/bin/bash
# t23 (card experiments/t23_lane_line_loss/experiment.md), arm C only: behind the lane project (D57), inside Mirko's GPU windows.
# chain4 did the smoke (read by hand), this one waits and then runs the real round:
#   gate : the lane chat's flag ../bev_fusion_lane_center/temp/e276/CHAIN_DONE exists (the lane chat holds both GPU slots with E279 until then;
#          its request of 2026-11-08 03:4x), a window with >= 140 min left, a free GPU (no train / eval / scoring process, < 2.6 GB used)
#   1 t23.C_s0     arm C (loss.line_bce 10), 1500 steps, then t23.C_s0.eval (evaluate.py, VAL 1000)
# The comparison against runs/t21_ctl_s0 is read by hand afterwards. No shutdown here.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
LOG=temp/lane_tests/chain5.log
exec >> $LOG 2>&1
PY=$HOME/miniconda3/envs/gcar/bin/python
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|bev_suite|qgraph_|trainsuite|run_ar\.sh" | grep -v $$ > /dev/null
}
# the lane chat's chain (lane first, D57): while its READY flag exists and CHAIN_DONE does not, this chain holds; the file temp/lane_tests/NOHOLD lifts the hold
E276=../bev_fusion_lane_center/temp/e276
e276_free() { [ -f temp/lane_tests/NOHOLD ] || [ ! -f $E276/READY ] || [ -f $E276/CHAIN_DONE ]; }
wait_gates() {  # $1 = minutes of window needed
  until [ ! -f temp/lane_tests/HOLD5 ] && e276_free && [ "$(window_left)" -ge "$1" ] && gpu_free; do sleep 30; done   # HOLD5 = a file anyone can touch to keep this chain waiting
  echo "gates open: window left $(window_left) min, e276 READY $([ -f $E276/READY ] && echo yes || echo no), CHAIN_DONE $([ -f $E276/CHAIN_DONE ] && echo yes || echo no): $(date)"
}
compose() {  # $1 = base config, $2 = id, rest = trainsuite args; writes temp/compose/<id>.json from the resolved preview (trainsuite names the preview with dots turned into underscores)
  id=$2; base=$1; shift 2
  ( cd ../autoresearch && python3 -m trainsuite start ../bev_fusion/training/configs/$base --id $id "$@" ) || return 1
  cp temp/compose/${id//./_}.preview.json temp/compose/$id.json
}
final_status() { python3 -c "import json,sys; print(json.load(open('records/runs/$1/run.json')).get('status'))" 2>/dev/null; }

echo "chain5 waits: $(date)"
wait_gates 140

arm() {  # $1 = arm letter, $2 = config file, $3 = hypothesis sentence
  A=$1; CFG=$2
  compose $CFG t23.${A}_s0 --no-card || { echo "compose $A failed"; return 1; }
  python3 ../autoresearch/tools/run.py --id t23.${A}_s0 --gpu --gpu-max-mb 3000 --split train --retries 0 \
    --hyp "$3" \
    --pred "VAL 1000, dt 0, against t21.ctl_s0 (card experiments/t23_lane_line_loss): lane divider IoU@0.5 0.4341 -> >= 0.455, IoU@1 0.5516 -> >= 0.565, recall at 0.5 m >= 0.55, lane divider AP and roc_auc not below the control by more than their half-widths; guards (road divider IoU@1, drivable, walkway, crossing, vehicle IoU dt0, the 1 m centre-line rows) not below the control by more than m = max(0.010, half-width), mAP fusion area by max(0.020, m), moving-centre velocity EPE not above +0.05 m/s" \
    --rule "by hand: composed from the table rows (card section 1): CANDIDATE if lane divider IoU@0.5 above the control by more than max(0.015, m) with the paired interval excluding 0, every guard holds, the ranking (AP, roc_auc) is not worse and the step-1 check passed (then a second seed of arm and control plus a lines-scale control); TIE inside +-m (provisional for one seed, for these settings only); CUT below -m with the interval below 0 or a guard broken by more than 2 m with the interval beyond -m; else INCONCLUSIVE (second seed)" \
    -- bash temp/train/run_ar.sh temp/compose/t23.${A}_s0.json
  echo "train $A exit $? status $(final_status t23.${A}_s0) $(date)"
  $PY temp/lane_tests/check_step1.py runs/t23_${A}_s0/log.jsonl runs/t21_ctl_s0/log.jsonl
  echo "step-1 check on the run: exit $?"
  [ "$(final_status t23.${A}_s0)" = "ok" ] || return 1
  python3 ../autoresearch/tools/run.py --id t23.${A}_s0.eval --gpu --gpu-max-mb 3000 --split val --no-card --retries 0 -- \
    bash -c "$PY -u evaluation/evaluate.py runs/t23_${A}_s0 --quant --curves --fusion --velocity --n 1000 >> runs/t23_${A}_s0_eval.log 2>&1"
  echo "eval $A exit $? status $(final_status t23.${A}_s0.eval) $(date)"
}

arm C t23_lane_bce.json "t23 arm C: a tolerance-aware BCE on the lane / road divider presence (loss.line_bce 10, positives x3, 1 m around a GT line ignored) added to the lines group of the main head in the 1500-step fine-tune of t20.base_s0 punishes the confident far false cells the Dice ignores and puts the 0.5 threshold near the IoU optimum, so lane divider IoU@0.5 rises over t21.ctl_s0"

# CPU read-out of the result for whoever comes next (no verdict: the card's rule is applied by hand): the paired bootstrap against the control and the prediction / guard table
if [ -f runs/t23_C_s0/eval/card.json ]; then
  $PY evaluation/bev_suite.py compare runs/t21_ctl_s0 runs/t23_C_s0 --curves --fusion --n 1000 > temp/lane_tests/t23_compare.txt 2>&1
  $PY temp/lane_tests/read_t23.py t23_C_s0 t21_ctl_s0 > temp/lane_tests/t23_readout.txt 2>&1
  echo "read-out written: temp/lane_tests/t23_readout.txt, t23_compare.txt: $(date)"
fi
touch temp/lane_tests/DONE5
echo "chain5 done $(date)"

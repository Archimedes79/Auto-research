#!/bin/bash
# t23 second seed (card experiments/t23_lane_line_loss, section 4), decided by Mirko ("Yes, tonight", 2026-11-08 evening).
#   1 t21.ctl_s1   control, seed 1 (current working tree), then t21.ctl_s1.eval
#   2 t23.C_s1     arm C, seed 1, then step-1 check against t21.ctl_s1, then t23.C_s1.eval
#   3 CPU read-out of the pair (compare + table), no verdict (the card's rule is applied by hand)
# Gates before each training (inside Mirko's windows, behind the others):
#   no HOLD5 file (anyone can touch it), the lane chat's CHAIN_DONE (or no READY), the Image Encoder's night chain temp/t22/night_B2.sh has ended
#   (Mirko's order: B2 first), >= 140 min of window left, a free GPU (no train / eval / scoring process, < 2.6 GB used); run.py --gpu-max-mb 3000 checks again.
# No shutdown here. If the control fails, the arm is not started.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
LOG=temp/lane_tests/chain7.log
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
E276=../bev_fusion_lane_center/temp/e276
e276_free() { [ -f temp/lane_tests/NOHOLD ] || [ ! -f $E276/READY ] || [ -f $E276/CHAIN_DONE ]; }
b2_over() { ! pgrep -f "t22/night_B2\.sh" > /dev/null; }
wait_gates() {  # $1 = minutes of window needed
  until [ ! -f temp/lane_tests/HOLD5 ] && e276_free && b2_over && [ "$(window_left)" -ge "$1" ] && gpu_free; do sleep 30; done
  sleep 20   # a second look: another chat's job may have started in the gap
  until [ ! -f temp/lane_tests/HOLD5 ] && e276_free && b2_over && [ "$(window_left)" -ge "$1" ] && gpu_free; do sleep 30; done
  echo "gates open: window left $(window_left) min, B2 chain over, e276 CHAIN_DONE $([ -f $E276/CHAIN_DONE ] && echo yes || echo no): $(date)"
}
compose() {  # $1 = base config, $2 = id, rest = trainsuite args; writes temp/compose/<id>.json from the resolved preview
  id=$2; base=$1; shift 2
  ( cd ../autoresearch && python3 -m trainsuite start ../bev_fusion/training/configs/$base --id $id --set seed=1 "$@" ) || return 1
  cp temp/compose/${id//./_}.preview.json temp/compose/$id.json
}
final_status() { python3 -c "import json,sys; print(json.load(open('records/runs/$1/run.json')).get('status'))" 2>/dev/null; }
eval_run() {  # $1 = run id with dots, $2 = run dir name
  python3 ../autoresearch/tools/run.py --id $1.eval --gpu --gpu-max-mb 3000 --split val --no-card --retries 0 -- \
    bash -c "$PY -u evaluation/evaluate.py runs/$2 --quant --curves --fusion --velocity --n 1000 >> runs/${2}_eval.log 2>&1"
  echo "eval $1 exit $? status $(final_status $1.eval) $(date)"
}

echo "chain7 waits: $(date)"

# ---- 1 control, seed 1
wait_gates 140
compose t21_ctl.json t21.ctl_s1 --no-card || { echo "compose ctl_s1 failed"; exit 1; }
python3 ../autoresearch/tools/run.py --id t21.ctl_s1 --gpu --gpu-max-mb 3000 --split train --retries 0 --seed 1 \
  --hyp "t21 control, second seed (seed 1) for t23 arm C: the equal-budget 1500-step fine-tune of t20.base_s0 without line_bce, run on the current working tree; it is the matched partner of t23.C_s1 and gives the spread of the control against t21.ctl_s0" \
  --pred "VAL 1000, dt 0 (card experiments/t23_lane_line_loss section 4): lane divider IoU@0.5 within 0.010 of t21.ctl_s0 (0.4341), AP and roc_auc within their half-widths of it; no verdict of its own" \
  --rule "by hand: no verdict of its own; read only as the partner of t23.C_s1 and as the spread of the control (card section 4)" \
  -- bash temp/train/run_ar.sh temp/compose/t21.ctl_s1.json
echo "train ctl_s1 exit $? status $(final_status t21.ctl_s1) $(date)"
if [ "$(final_status t21.ctl_s1)" != "ok" ]; then echo "control failed: arm not started"; touch temp/lane_tests/DONE7; exit 1; fi
eval_run t21.ctl_s1 t21_ctl_s1

# ---- 2 arm C, seed 1
wait_gates 140
compose t23_lane_bce.json t23.C_s1 --no-card || { echo "compose C_s1 failed"; exit 1; }
python3 ../autoresearch/tools/run.py --id t23.C_s1 --gpu --gpu-max-mb 3000 --split train --retries 0 --seed 1 \
  --hyp "t23 arm C, second seed (seed 1): a tolerance-aware BCE on the lane / road divider presence (loss.line_bce 10, positives x3, 1 m around a GT line ignored) in the 1500-step fine-tune of t20.base_s0; seed 0 gave a TIE in IoU@0.5 with a clearly better ranking (AP +0.038); does this reproduce against the matched control t21.ctl_s1" \
  --pred "VAL 1000, dt 0, against t21.ctl_s1 (card experiments/t23_lane_line_loss section 4): d1 = lane divider IoU@0.5 difference inside +-0.010 (TIE again); lane divider AP difference >= +0.020 and roc_auc difference >= +0.030; spread of the arm |C_s1 - C_s0| <= 0.010 in IoU@0.5" \
  --rule "by hand: composed from the table rows (card section 4): CANDIDATE only if d1 > max(0.015, m1) with the interval excluding 0, (d0 + d1) / 2 > +0.010, every guard holds in both seeds and AP / roc_auc are not worse in both; TIE on two seeds if |d1| <= m1 (m1 = max(0.010, half-width)); CUT if d1 < -m1 with the interval below 0; else INCONCLUSIVE (the seeds disagree)" \
  -- bash temp/train/run_ar.sh temp/compose/t23.C_s1.json
echo "train C_s1 exit $? status $(final_status t23.C_s1) $(date)"
$PY temp/lane_tests/check_step1.py runs/t23_C_s1/log.jsonl runs/t21_ctl_s1/log.jsonl
echo "step-1 check on the run: exit $?"
if [ "$(final_status t23.C_s1)" = "ok" ]; then eval_run t23.C_s1 t23_C_s1; fi

# ---- 3 CPU read-out of the pair (no verdict)
if [ -f runs/t23_C_s1/eval/card.json ] && [ -f runs/t21_ctl_s1/eval/card.json ]; then
  $PY evaluation/bev_suite.py compare runs/t21_ctl_s1 runs/t23_C_s1 --curves --fusion --n 1000 > temp/lane_tests/t23_s1_compare.txt 2>&1
  $PY temp/lane_tests/read_t23.py t23_C_s1 t21_ctl_s1 > temp/lane_tests/t23_s1_readout.txt 2>&1
  echo "read-out written: temp/lane_tests/t23_s1_readout.txt, t23_s1_compare.txt: $(date)"
fi
touch temp/lane_tests/DONE7
echo "chain7 done $(date)"

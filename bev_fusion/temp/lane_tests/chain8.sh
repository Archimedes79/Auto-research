#!/bin/bash
# t24 (card experiments/t24_lane_pull/experiment.md): the lane branch lifts with lift P instead of lift PD; two seeds, each against its control (t21.ctl_s0 stored, t21.ctl_s1 from chain7).
#   0 smoke.t24_ctlcode: 20 steps of t21_ctl.json on the working tree, row 1 against t21.ctl_s0 (code equivalence, read by hand)
#   1 t24.P_s0 + eval + compare against runs/t21_ctl_s0
#   2 t24.P_s1 + eval + compare against runs/t21_ctl_s1
# ON HOLD behind the card's go / no-go: the chain waits for the flag file temp/lane_tests/GO8, written by hand only if the L4 read-out says L4 contributes.
# Gates before each training: DAYTIME window only (Monday - Friday 08:30 - 17:30): the night windows belong to the Image Encoder's B2 chain and its L4 check
#   (Mirko picked B2 first); the lane chat's CHAIN_DONE (or no READY); no HOLD5 file (anyone can touch it: lane first, D57); chain7 finished (flag DONE7);
#   a free GPU (no train / eval / scoring process, < 2.6 GB used, two looks 20 s apart); >= 140 min of the window left. run.py --gpu-max-mb 3000 checks again.
# The CPU read-outs (temp/overview/lane_readout.py: IoU outside the intersections, recall with / without paint, thresholds) are NOT part of this chain: CPU only, run by hand.
# No shutdown here. If a seed's training fails, the chain stops (the pair is useless).
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
LOG=temp/lane_tests/chain8.log
exec >> $LOG 2>&1
PY=$HOME/miniconda3/envs/gcar/bin/python
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
day_left() {   # minutes left of the Monday - Friday 08:30 - 17:30 window, 0 outside it
  d=$(date +%u); n=$(now_min)
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); else echo 0; fi
}
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|bev_suite|qgraph_|trainsuite|run_ar\.sh" | grep -v $$ > /dev/null
}
E276=../bev_fusion_lane_center/temp/e276
e276_free() { [ -f temp/lane_tests/NOHOLD ] || [ ! -f $E276/READY ] || [ -f $E276/CHAIN_DONE ]; }
open_now() { [ ! -f temp/lane_tests/HOLD5 ] && [ -f temp/lane_tests/GO8 ] && e276_free && [ -f temp/lane_tests/DONE7 ] && [ "$(day_left)" -ge "$1" ] && gpu_free; }   # GO8 = the go / no-go of the card (written by hand after the L4 read-out)
wait_gates() {  # $1 = minutes of window needed
  until open_now $1; do sleep 30; done
  sleep 20   # a second look: another chat's job may have started in the gap
  until open_now $1; do sleep 30; done
  echo "gates open: day window left $(day_left) min, DONE7 yes, CHAIN_DONE $([ -f $E276/CHAIN_DONE ] && echo yes || echo no): $(date)"
}
compose() {  # $1 = base config, $2 = id, $3 = seed, rest = more trainsuite args; writes temp/compose/<id>.json from the resolved preview
  base=$1; id=$2; seed=$3; shift 3
  ( cd ../autoresearch && python3 -m trainsuite start ../bev_fusion/training/configs/$base --id $id --set seed=$seed "$@" --no-card ) || return 1
  cp temp/compose/${id//./_}.preview.json temp/compose/$id.json
}
smoke() {   # 20 steps of t21_ctl.json on the working tree: its row 1 must equal the log of t21.ctl_s0 in EVERY term (code equivalence, card section 1); read by hand
  wait_gates 60
  compose t21_ctl.json smoke.t24_ctlcode 0 --set steps=20 || { echo "compose smoke failed"; return 1; }
  python3 ../autoresearch/tools/run.py --id smoke.t24_ctlcode --gpu --gpu-max-mb 3000 --split train --retries 0 --no-card \
    -- bash temp/train/run_ar.sh temp/compose/smoke.t24_ctlcode.json
  echo "smoke exit $? status $(final_status smoke.t24_ctlcode) $(date)"
  $PY temp/lane_tests/check_step1.py runs/smoke_t24_ctlcode/log.jsonl runs/t21_ctl_s0/log.jsonl
  echo "step-1 check of the smoke against t21.ctl_s0: exit $?"
}
final_status() { python3 -c "import json,sys; print(json.load(open('records/runs/$1/run.json')).get('status'))" 2>/dev/null; }

arm() {  # $1 = seed
  s=$1; id=t24.P_s$s; dir=t24_P_s$s; ctl=t21_ctl_s$s
  wait_gates 140
  compose t24_lane_pull.json $id $s || { echo "compose $id failed"; return 1; }
  python3 ../autoresearch/tools/run.py --id $id --gpu --gpu-max-mb 3000 --split train --retries 0 --seed $s \
    --hyp "t24 arm P seed $s: the stride-4 lane branch L4 lifts its features with lift P (pull on the same projection table, no depth weight, model.lane_use_depth false) instead of lift PD in the 1500-step fine-tune of t20.base_s0; the DualBEV depth probabilities were trained for objects at stride 16 and can only blur or suppress thin ground paint, so lane divider IoU@0.5 rises over the control of the same seed t21.ctl_s$s" \
    --pred "VAL 1000, dt 0, against t21.ctl_s$s (card experiments/t24_lane_pull; seed 0 control: lane divider IoU@0.5 0.4341, IoU@1 0.5516, AP 0.2149): IoU@0.5 at least +0.015, IoU@1 at least +0.010, lane divider AP and roc_auc not below the control by more than their half-widths; guards (road divider IoU@1, drivable, walkway, crossing, vehicle IoU dt0, the three lane-centre rows) not below by more than m = max(0.010, half-width), mAP fusion area by max(0.020, m), moving-centre velocity EPE not above +0.05 m/s" \
    --rule "by hand: composed from the table rows (card section 1, written before the run): over both seeds D = mean of d_s (IoU@0.5 arm minus control), M = mean of m_s = max(0.010, half-width); CANDIDATE if D > max(0.015, M), no seed has d_s < 0, one interval excludes 0, every guard holds in both seeds and AP / roc_auc are not worse in both; TIE if |D| <= M; CUT if D < -M with both d_s < 0, or a guard broken by more than 2 m in both seeds; else INCONCLUSIVE (the seeds disagree)" \
    -- bash temp/train/run_ar.sh temp/compose/$id.json
  echo "train $id exit $? status $(final_status $id) $(date)"
  [ "$(final_status $id)" = "ok" ] || return 1
  python3 ../autoresearch/tools/run.py --id $id.eval --gpu --gpu-max-mb 3000 --split val --no-card --retries 0 -- \
    bash -c "$PY -u evaluation/evaluate.py runs/$dir --quant --curves --fusion --velocity --n 1000 >> runs/${dir}_eval.log 2>&1"
  echo "eval $id exit $? status $(final_status $id.eval) $(date)"
  if [ -f runs/$dir/eval/card.json ] && [ -f runs/$ctl/eval/card.json ]; then
    $PY evaluation/bev_suite.py compare runs/$ctl runs/$dir --curves --fusion --velocity --n 1000 > temp/lane_tests/t24_compare_s$s.txt 2>&1
    $PY temp/lane_tests/read_t24.py $dir $ctl > temp/lane_tests/t24_readout_s$s.txt 2>&1
    echo "read-out s$s written: temp/lane_tests/t24_readout_s$s.txt, t24_compare_s$s.txt: $(date)"
  fi
}

echo "chain8 waits: $(date)"
smoke || echo "smoke problem: read the log"
arm 0 || { echo "arm 0 failed"; touch temp/lane_tests/DONE8; exit 1; }
arm 1 || { echo "arm 1 failed"; }
touch temp/lane_tests/DONE8
echo "chain8 done $(date)"

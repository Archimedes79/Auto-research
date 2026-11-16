#!/bin/bash
# E275 step 2: wait for Mirko's GPU window (Monday-Friday 08:30-17:30, every night 22:00-07:00) with enough time left and a free GPU, then score the four models one after the other through run.py
# (ledger rows E275.n_s0, E275.n_s1 = baseline B with the step; E275.rn_s0, E275.rn_s1 = arm r with the step, reported). One GPU slot at a time, 2 CPU threads each.
cd "$(dirname "$0")/../.."
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {                                   # minutes left in the current window (0 = outside)
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|qgraph_train|trainsuite|qgraph_run_eval" | grep -v $$ > /dev/null
}
for m in e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1; do [ -f temp/e274/sweep_$m.json ] || { echo "missing temp/e274/sweep_$m.json: stop before waiting for the window"; exit 2; }; done   # the first night was lost to a bad sweep path
NEED=${1:-150}                                   # minutes: four models, about 25 + 25 + 12 + 12 min of wall clock, with margin
until [ -f temp/e275/READY ] && [ "$(window_left)" -ge "$NEED" ] && gpu_free; do sleep 30; done
echo "start $(date), window left $(window_left) min"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
HYP="E275: dropping every delivered node that lies within 1.5 m of a node with a higher existence probability (a named read-out step, no retraining) lifts the delivered graph of the E273 baseline: topo F1 and junction F1 above the same checkpoints read out as before, with the assignment guards held"
PRED="topo F1 0.596 -> 0.626 (+0.03), junction F1 0.125 -> 0.26; exact-map assignment p-g >= -0.01 for the step on B and on r; >= 40 % of the dropped nodes inside intersections; < 3 % of true nodes have another true node within 1.5 m (already missed on the GT alone before this run: 32.3 %); junction F1 maximum at 1.25-1.75 m, collapse at 2.0 m; step < 1 ms per frame"
RULE="E275 card section 0: KEEP if topo F1 gain over c > +0.02 AND junction F1 gain > +0.03 on both seeds and guards hold (lane_acc_exact p-g >= -0.01, lane_acc_err p-g >= +0.03, F1@1.0 >= c - 0.01, lanes_found >= c - 0.03); TIE inside both bands; CUT on a negative gain or a broken guard; INCONCLUSIVE otherwise; by hand (verdict275); in-sample on VAL 873, no TEST"
RO_B="plain@0.5 close:1.5@0.5 close:1.5@0.4 plain@0.4"
RO_R="plain@0.5 close:1.5@0.5"
run1() {  # id model seed readouts extra...
  id=$1; model=$2; seed=$3; ro=$4; shift 4
  python3 ../autoresearch/tools/run.py --id $id --seed $seed --gpu --retries 0 --require-card $AFTER --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
    -- bash tools/env/_py14.sh qgraph_run_eval.py score --run $model --seed $seed --exp e275 --readouts $ro --repro temp/e274/sweep_$model.json "$@"
}
AFTER=""
run1 E275.n_s0 e273_p_s0 0 "$RO_B" --curve 1.0 1.25 1.75 2.0 --dump || exit 1
AFTER="--after E275.n_s0"
run1 E275.n_s1 e273_p_s1 1 "$RO_B" --curve 1.0 1.25 1.75 2.0 --dump || exit 1
AFTER="--after E275.n_s1"
run1 E275.rn_s0 e274_r_s0 0 "$RO_R" --dump || exit 1
AFTER="--after E275.rn_s0"
run1 E275.rn_s1 e274_r_s1 1 "$RO_R" --dump || exit 1
touch temp/e275/SCORED
echo "done $(date)"

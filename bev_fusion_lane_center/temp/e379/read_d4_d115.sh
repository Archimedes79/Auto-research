#!/bin/bash
# exploratory d4_s0 (KD weight 4, NOT a verdict arm): the VAL score at the FIXED D115 cell against the champion head (no TRAIN sweep: a cell chosen on its own TRAIN table is for later, after the quiet chain); waits for the recipe-cell read to end (one CPU job at a time)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
while pgrep -f "read_recipe.sh" > /dev/null; do sleep 10; done
source temp/e379/texts379.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
$RUN --id E379.d115_d4_s0 --seed 0 --retries 0 --no-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e379_d4_s0 --seed 0 --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s0 --ref-run-cell 0.3,0.2,0.6,off && $PY jtoken_eval.py anatomy --run e379_d4_s0 --seed 0 --link-thr-long 0.2 --keep-thr 0.6"
echo "$(date '+%m-%d %H:%M:%S') [read d4_s0 D115 (exploratory)] exit $?" >> temp/e379/chain379.log

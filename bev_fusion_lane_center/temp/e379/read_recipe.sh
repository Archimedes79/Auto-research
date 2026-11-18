#!/bin/bash
# E379 seed 0 at the RECIPE cell (the card's rule: the heads' own cells differ -> the cell of the MEAN of the two TRAIN tables, 0.3 / 0.2 / 0.5 / off, written to temp/e379/recipe_cell/ BEFORE this VAL score); d_s1's own pinned cell was the same one, so its read exists (E379.read_d_s1)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e379/texts379.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
$RUN --id E379.recipe_d_s0 --seed 0 --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e379_d_s0 --seed 0 --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.5 --no-none-gate --ref-run e373_cb_s0 --ref-run-cell 0.3,0.2,0.6,off"
echo "$(date '+%m-%d %H:%M:%S') [read d_s0 recipe cell] exit $?" >> temp/e379/chain379.log

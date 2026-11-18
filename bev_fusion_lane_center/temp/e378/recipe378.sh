#!/bin/bash
# E378 (card v2, read-outs: "if the two heads pick different cells ... the verdict reads both heads at that recipe cell, E378.recipe_x_s<S>"):
# the recipe cell 0.3,0.2,0.5,off (choose378.py e378_x_s0 e378_x_s1, logged 04:51 before any VAL score at it) = x_s1's own cell (its E378.read_x_s1 is that read);
# x_s0 is read at it here (CPU), against the champion head of the same cache at D115, as read378.sh reads.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e378/texts378.sh
say() { echo "$(date '+%m-%d %H:%M:%S') [recipe x_s0] $*" >> temp/e378/chain378.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
python3 ../autoresearch/tools/run.py --id E378.recipe_x_s0 --seed 0 --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e378_x_s0 --seed 0 --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.5 --no-none-gate --ref-run e373_cb_s0 --ref-run-cell 0.3,0.2,0.6,off" >> temp/e378/read_recipe_x_s0.out 2>&1
say "exit $?"

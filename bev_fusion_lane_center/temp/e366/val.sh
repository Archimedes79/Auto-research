#!/bin/bash
# E366: the chosen cell applied ONCE to VAL for one seed (CPU, niced). usage: bash temp/e366/val.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e365_s1_s$S
source temp/e366/texts.sh
python3 ../autoresearch/tools/run.py --id E366.val_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S --link-thr 0.1 --keep-thr 0.6 --no-none-gate >> temp/e366/val_s$S.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [e366 s$S] val exit $?" >> temp/e366/chain.log

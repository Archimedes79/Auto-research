#!/bin/bash
# E375 per-class sweep on the TRAIN sample of one E367 b_bal25 head (CPU, niced). usage: bash temp/e375/sweep.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}
source temp/e375/texts.sh
python3 ../autoresearch/tools/run.py --id E375.sweep_s$S --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run e367_b25_s$S --seed $S --split train --every 3 --grid perclass >> temp/e375/sweep_s$S.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [sweep s$S] exit $?" >> temp/e375/chain.log

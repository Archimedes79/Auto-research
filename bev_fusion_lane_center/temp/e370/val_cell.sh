#!/bin/bash
# E370 stage-B heads at their TRAIN-selected cell (0.1 / 0.6 / off) on VAL: a diagnosis for the cross-card comparison (never the E370 verdict). usage: bash temp/e370/val_cell.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}
source temp/e370/texts.sh
python3 ../autoresearch/tools/run.py --id E370.cell_b_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --run e370_b_s$S --seed $S --link-thr 0.1 --keep-thr 0.6 --no-none-gate >> temp/e370/cell_s$S.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [cell_b s$S] exit $?" >> temp/e370/chain.log

#!/bin/bash
# E370 stage-A fixture per class on the TRAIN sample (CPU, niced), through run.py. usage: bash temp/e370/fixture.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}
source temp/e370/texts.sh
python3 ../autoresearch/tools/run.py --id E370.fixa_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py fixture --seed $S --run e370_a_s$S --split train --every 3 >> temp/e370/fixture_s$S.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [fixture a_s$S] exit $?" >> temp/e370/chain.log

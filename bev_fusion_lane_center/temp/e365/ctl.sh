#!/bin/bash
# E365 controls (CPU, niced): c-hi, c-rule, c-floor, c-B-sec0 for B' seed S, one run.py job per seed; usage: bash temp/e365/ctl.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=${1:?seed}
source temp/e365/texts.sh
python3 ../autoresearch/tools/run.py --id E365.ctl_s$S --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- nice -n 10 bash tools/env/_py14.sh jtoken_eval.py ctl --seed $S --ident e292_a >> temp/e365/ctl_s$S.out 2>&1
echo "ctl s$S exit $?" >> temp/e365/chain.log

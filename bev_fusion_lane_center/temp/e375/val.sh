#!/bin/bash
# E375: the chosen per-class cell scored ONCE on VAL for one head, with the champion's cell as the paired reference (CPU, niced). usage: bash temp/e375/val.sh <run id suffix> <run> <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ID=${1:?}; R=${2:?}; S=${3:?}
source temp/e375/texts.sh
python3 ../autoresearch/tools/run.py --id E375.$ID --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-cell 0.2,0.2,0.6,off >> temp/e375/val_$ID.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [val $ID] exit $?" >> temp/e375/chain.log

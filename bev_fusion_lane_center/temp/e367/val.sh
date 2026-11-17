#!/bin/bash
# E367: one head scored ONCE on VAL at its chosen cell (CPU, niced) + the degraded read-out at the same cell. usage: bash temp/e367/val.sh <id> <seed> <link> <keep> <gate on|off>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ID=${1:?}; S=${2:?}; LT=${3:?}; KT=${4:?}; G=${5:?}; R=e367_$ID
NG=""; [ "$G" = off ] && NG="--no-none-gate"
source temp/e367/texts.sh
python3 ../autoresearch/tools/run.py --id E367.val_$ID --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash -c "bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S --link-thr $LT --keep-thr $KT $NG && bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S && bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S --drop 0.17 --drop-seed 0 --link-thr $LT --keep-thr $KT $NG" >> temp/e367/val_$ID.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [val $ID] exit $?" >> temp/e367/chain.log

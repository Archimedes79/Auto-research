#!/bin/bash
# E370 stage-A VAL read-out (B-ADDS reference): dump (GPU ticket, FIFO) + score at the decode of record (CPU). usage: bash temp/e370/read_a.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e370_a_s$S
source temp/e370/texts.sh
python3 ../autoresearch/tools/run.py --id E370.read_a_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 40 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split val && env OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --run $R --seed $S" >> temp/e370/read_a_s$S.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') [read_a s$S] exit $?" >> temp/e370/chain.log

#!/bin/bash
# E373 read-out of one fresh head: waits for the training, new-format dumps (GPU), VAL scores at D-rec and D-cell (0.2 / 0.6 / off), the degraded read-out at both, then the TRAIN sample (diagnosis). usage: bash temp/e373/read.sh <cache seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e373_cb_s$S
source temp/e373/texts.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read cb_s$S] $*" >> temp/e373/chain.log; }
until grep -q "^[0-9:]* done $R\$" runs/$R/log.txt 2>/dev/null; do sleep 30; done
PY="env OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"; CELL="--link-thr 0.2 --keep-thr 0.6 --no-none-gate"
$RUN --id E373.read_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split val --drop 0.17 --drop-seed 0 && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3 && $PY jtoken_eval.py score --run $R --seed $S && $PY jtoken_eval.py score --run $R --seed $S $CELL && $PY jtoken_eval.py score --run $R --seed $S --drop 0.17 --drop-seed 0 && $PY jtoken_eval.py score --run $R --seed $S --drop 0.17 --drop-seed 0 $CELL && $PY jtoken_eval.py score --run $R --seed $S --split train --every 3 $CELL" >> temp/e373/read_s$S.out 2>&1
say "read exit $?"

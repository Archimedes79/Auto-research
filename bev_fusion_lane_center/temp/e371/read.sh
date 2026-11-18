#!/bin/bash
# E371 read-out of one fresh head: waits for the training, then new-format dumps (GPU), the VAL score at the FROZEN decode (0.1 / 0.6 / gate off) and the record,
# the TRAIN-sample score at the frozen decode (diagnosis), then the TRAIN sweep (diagnosis, after the VAL score). usage: bash temp/e371/read.sh <cache seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e371_cf_s$S
source temp/e371/texts.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read cf_s$S] $*" >> temp/e371/chain.log; }
until grep -q "^[0-9:]* done $R\$" runs/$R/log.txt 2>/dev/null; do sleep 30; done
PY="bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
$RUN --id E371.read_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3 && env OMP_NUM_THREADS=2 nice -n 10 $PY jtoken_eval.py score --run $R --seed $S --link-thr 0.1 --keep-thr 0.6 --no-none-gate && env OMP_NUM_THREADS=2 nice -n 10 $PY jtoken_eval.py score --run $R --seed $S" >> temp/e371/read_s$S.out 2>&1
say "read exit $?"
$RUN --id E371.diag_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash -c "$PY jtoken_eval.py score --run $R --seed $S --split train --every 3 --link-thr 0.1 --keep-thr 0.6 --no-none-gate && $PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3" >> temp/e371/read_s$S.out 2>&1
say "diag exit $?"

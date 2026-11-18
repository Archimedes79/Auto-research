#!/bin/bash
# E370 stage-B read-out of one seed: waits for the training; GPU dumps (VAL, VAL degraded, TRAIN every 3, and the TRAIN sample of the snapshots 0 / 500 / 1000 / 3000 / 6000),
# then on the CPU: the VAL score at the decode of record (the verdict), the degraded read-out, the TRAIN-sample scores of the snapshots and of the final head (the drift), the TRAIN sweep (the decode for the cross-card choice).
# usage: bash temp/e370/read_b.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e370_b_s$S
source temp/e370/texts.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read_b s$S] $*" >> temp/e370/chain.log; }
until grep -q "^[0-9:]* done $R\$" runs/$R/log.txt 2>/dev/null; do sleep 30; done
PYR="bash tools/env/_pyr.sh"; PY="env OMP_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
D="$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split val --drop 0.17 --drop-seed 0 && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3"
for N in 0 500 1000 3000 6000; do D="$D && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3 --model model_step$N.pt"; done
$RUN --id E370.dump_b_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 40 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$D" >> temp/e370/read_b_s$S.out 2>&1
say "dumps exit $?"
C="$PY jtoken_eval.py score --run $R --seed $S && $PY jtoken_eval.py score --run $R --seed $S --drop 0.17 --drop-seed 0 && $PY jtoken_eval.py score --run $R --seed $S --split train --every 3"
for N in 0 500 1000 3000 6000; do C="$C && $PY jtoken_eval.py score --run $R --seed $S --split train --every 3 --model model_step$N.pt"; done
C="$C && $PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3"
$RUN --id E370.read_b_s$S --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$C" >> temp/e370/read_b_s$S.out 2>&1
say "read exit $?"

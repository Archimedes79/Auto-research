#!/bin/bash
# E367 read-out preparation for one head: waits until the training is done, then new-format dumps (TRAIN every 3, VAL, VAL degraded) on the GPU, then the TRAIN sweep on the CPU;
# the cell is chosen by hand by the rule of E366 and the VAL score follows. usage: bash temp/e367/prep.sh <id> <seed>   e.g. b5_s0 0
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ID=${1:?id}; S=${2:?seed}; R=e367_$ID
source temp/e367/texts.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [prep $ID] $*" >> temp/e367/chain.log; }
until grep -q "^[0-9:]* done $R\$" runs/$R/log.txt 2>/dev/null; do sleep 60; done
say "training done, dumps"
$RUN --id E367.dump_$ID --seed $S --gpu --retries 1 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split train --every 3 && bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split val && bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split val --drop 0.17 --drop-seed 0" >> temp/e367/prep_$ID.out 2>&1
rc=$?; say "dumps exit $rc"; [ $rc = 0 ] || exit 1
$RUN --id E367.sweep_$ID --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run $R --seed $S --split train --every 3 >> temp/e367/prep_$ID.out 2>&1
say "sweep exit $?"

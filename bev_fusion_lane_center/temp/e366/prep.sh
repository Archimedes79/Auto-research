#!/bin/bash
# E366 preparation for one seed: new-format dumps of the S1 head (TRAIN every 3, VAL) on the GPU, then the TRAIN sweep on the CPU; the choice of the cell and the VAL score follow by hand. usage: bash temp/e366/prep.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
S=${1:?seed}; R=e365_s1_s$S
source temp/e366/texts.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e366 s$S] $*" >> temp/e366/chain.log; }
$RUN --id E366.dump_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split train --every 3 && bash tools/env/_pyr.sh jtoken_eval.py dump --run $R --seed $S --split val" >> temp/e366/prep_s$S.out 2>&1
rc=$?; say "dumps exit $rc"; [ $rc = 0 ] || exit 1
$RUN --id E366.sweep_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run $R --seed $S --split train --every 3 >> temp/e366/prep_s$S.out 2>&1
say "sweep exit $?"

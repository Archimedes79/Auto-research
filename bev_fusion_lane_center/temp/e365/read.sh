#!/bin/bash
# E365 read-out of one seed through run.py: dump + score on VAL, the TRAIN sample, the degraded read-out (17 % of the tokens removed). usage: bash temp/e365/read.sh <seed>
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=${1:?seed}
source temp/e365/texts.sh
R=e365_s1_s$S; PY="bash tools/env/_py14.sh"
CMD="$PY jtoken_eval.py dump --run $R --seed $S && $PY jtoken_eval.py score --run $R --seed $S && $PY jtoken_eval.py dump --run $R --seed $S --split train && $PY jtoken_eval.py score --run $R --seed $S --split train && $PY jtoken_eval.py dump --run $R --seed $S --drop 0.17 --drop-seed 0 && $PY jtoken_eval.py score --run $R --seed $S --drop 0.17 --drop-seed 0"
python3 ../autoresearch/tools/run.py --id E365.read_s$S --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$CMD" >> temp/e365/read_s$S.out 2>&1
echo "read s$S exit $?" >> temp/e365/chain.log

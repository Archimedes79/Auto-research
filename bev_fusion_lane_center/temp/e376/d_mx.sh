#!/bin/bash
# E376 read-out (d): the mx heads at the single-threshold cell 0.2 / 0.2 / 0.6 / off (written in the card before this score), against the champion heads at the D115 cell (CPU, run.py)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e376/texts376.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E376.d_mx_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e376_mx_s$S --seed $S --link-thr 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off" >> temp/e376/d_mx_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [d mx_s$S] exit $?" >> temp/e376/chain376.log
done

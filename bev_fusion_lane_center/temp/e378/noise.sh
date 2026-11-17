#!/bin/bash
# E378 (card v2, critic S5 / S6): the noise reference stored as cards: the champion recipe on another head seed (E367 b25) minus the E373 champion heads at the D115 cell (CPU, run.py)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e378/texts378.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E378.noise_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e367_b25_s$S --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s$S" >> temp/e378/noise_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [noise s$S] exit $?" >> temp/e378/chain378.log
done

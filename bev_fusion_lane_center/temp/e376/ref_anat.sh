#!/bin/bash
# E376 read-out: the anatomy (long-link categories, APs, concentration, near / far table) of the reference heads e373_cb_s0 / s1 at the D115 cell, for the side-by-side with the mx heads (CPU, run.py)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e376/texts376.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E376.ref_anat_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py anatomy --run e373_cb_s$S --seed $S --link-thr-long 0.2 --keep-thr 0.6" >> temp/e376/ref_anat_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [ref anat s$S] exit $?" >> temp/e376/chain376.log
done

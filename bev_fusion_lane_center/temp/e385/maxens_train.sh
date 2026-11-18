#!/bin/bash
# E385 v2.1 (second look): the max-head ensemble on the TRAIN sample (decides) against BOTH heads, and VAL against the second head (reported); seed 0 (seed 1 when x_s1 exists)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
C="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
for R in e376_mx_s0 e378_x_s0; do
  $PY jtoken_eval.py ensemble --runs e376_mx_s0,e378_x_s0 --seed 0 --split train --every 3 $C --ref-run $R --out-dir ../temp/e385 >> temp/e385/maxens_train_s0.out 2>&1; echo "train vs $R exit $?"
done
$PY jtoken_eval.py ensemble --runs e376_mx_s0,e378_x_s0 --seed 0 $C --ref-run e378_x_s0 --out-dir ../temp/e385 >> temp/e385/maxens_s0_vsx.out 2>&1; echo "val vs x exit $?"

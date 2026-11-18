#!/bin/bash
# E385 v2 (critic B2): is the ensemble of two MAX-positive heads better than one max head? (CPU, the committed ensemble command, out-dir temp/e385)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py ensemble --runs e376_mx_s0,e378_x_s0 --seed 0 --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e376_mx_s0 --out-dir ../temp/e385 > temp/e385/maxens_s0.out 2>&1
echo "exit $?"

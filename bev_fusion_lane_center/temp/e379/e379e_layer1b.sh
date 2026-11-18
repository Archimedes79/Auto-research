#!/bin/bash
# E379 layer 1 of the second dev copy of `jtoken_eval.py ensemble` (the floor on the card): D115 cell on VAL, no --ref-run; cards to temp/e379/e379e_layer1b/ (never runs/)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do
  nice -n 10 bash tools/env/_py14.sh temp/e379/e379e_session_scratch_dev2/evaluation/jtoken_eval.py ensemble --runs e373_cb_s$s,e367_b25_s$s --seed $s \
    --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --out-dir ../../e379e_layer1b > temp/e379/e379e_layer1b_s$s.out 2>&1
done

#!/bin/bash
# U5 scratch: test_jtoken (fixed refusal table and teachers index) and the full test_jlive_train (selftest, real check, mutants) on the edited jlive_train of the mirror tree
cd "$(dirname "$0")/../.."
export U2_WT=/mnt/c/Auto-research_u5_tree/bev_fusion_lane_center CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in training/grid/test_jtoken.py temp/simplify/u_jlive_check.py; do
  echo "=== $s"
  nice -n 10 bash tools/env/_py14.sh ux_run_with.py $s > temp/simplify/u5b_$(basename $s .py).out 2>&1
  echo "rc $?"
done
echo "u5b done"

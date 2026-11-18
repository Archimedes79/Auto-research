#!/bin/bash
# U2 scratch: the speed path with the forced adaptations of temp/simplify/u2_*fast* / u2_bench_token_step.py (NOT in the worktree: the lead decides) against the U2 modules of the worktree; and the diff of those adaptations
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=$(pwd)/temp/simplify
export U2_EXTRA="jtoken_fast=$S/u2_jtoken_fast.py;jtoken_fast_train=$S/u2_jtoken_fast_train.py" U2_SRC="$S/u2_test_jtoken_fast.py"
nice -n 10 bash tools/env/_py14.sh u2_run_with.py training/grid/test_jtoken_fast.py > temp/simplify/u2_fast_check.out 2>&1
echo "rc $?" >> temp/simplify/u2_fast_check.out
W=../../Auto-research_wt_simplify/bev_fusion_lane_center
: > temp/simplify/u2_fast_path_forced.diff
for p in training/grid/jtoken_fast.py:u2_jtoken_fast.py training/train/jtoken_fast_train.py:u2_jtoken_fast_train.py training/grid/test_jtoken_fast.py:u2_test_jtoken_fast.py training/train/bench_token_step.py:u2_bench_token_step.py; do
  diff -u --label "a/bev_fusion_lane_center/${p%%:*}" --label "b/bev_fusion_lane_center/${p%%:*}" "$W/${p%%:*}" "temp/simplify/${p##*:}" >> temp/simplify/u2_fast_path_forced.diff
done
echo "u2_fast_check done"

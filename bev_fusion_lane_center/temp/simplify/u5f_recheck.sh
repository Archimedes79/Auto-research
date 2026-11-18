#!/bin/bash
# U5 scratch (side check, not part of U5): the patched speed path again after the live_inputs assert was put back into u5f_jtoken_fast_train.py; the diff remade
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 U2_WT=/mnt/c/Auto-research_u5_tree/bev_fusion_lane_center
S=$(pwd)/temp/simplify
U2_EXTRA="jtoken_fast=$S/u5f_jtoken_fast.py;jtoken_fast_train=$S/u5f_jtoken_fast_train.py" U2_SRC="$S/u5f_test_jtoken_fast.py" nice -n 10 bash tools/env/_py14.sh ux_run_with.py training/grid/test_jtoken_fast.py > $S/u5f_patched.out 2>&1; echo "patched rc $?"; grep -v "^u2_run" $S/u5f_patched.out | tail -n 3 | cut -c1-300
W=/mnt/c/Auto-research_wt_simplify/bev_fusion_lane_center
: > $S/u5f_fast_path_forced.diff
for p in training/grid/jtoken_fast.py:u5f_jtoken_fast.py training/train/jtoken_fast_train.py:u5f_jtoken_fast_train.py training/grid/test_jtoken_fast.py:u5f_test_jtoken_fast.py training/train/bench_token_step.py:u5f_bench_token_step.py; do
  diff -u --label "a/bev_fusion_lane_center/${p%%:*}" --label "b/bev_fusion_lane_center/${p%%:*}" "$W/${p%%:*}" "$S/${p##*:}" >> $S/u5f_fast_path_forced.diff
done
echo "diff +$(grep -c '^+[^+]' $S/u5f_fast_path_forced.diff) -$(grep -c '^-[^-]' $S/u5f_fast_path_forced.diff)"

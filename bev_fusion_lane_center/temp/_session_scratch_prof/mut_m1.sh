#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
f=training/grid/jtoken_fast.py
cp $f temp/_session_scratch_prof/jtoken_fast.bak
sed -i 's/ \& cand\[:, 1\]\[bi, ii, jj\]//' $f
grep -c "cand\[:, 1\]\[bi, ii, jj\]" $f
CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh training/grid/test_jtoken_fast.py sparse 2>&1 | tail -3
cp temp/_session_scratch_prof/jtoken_fast.bak $f
grep -c "cand\[:, 1\]\[bi, ii, jj\]" $f

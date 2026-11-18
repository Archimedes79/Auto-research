#!/bin/bash
# E358 scratch: experiments/E358_wiring-head-on-attached-ends/staged_diff.patch = diff -u of each staged file against its original (tree), one file
cd /mnt/c/Auto-research/bev_fusion_lane_center
S=temp/_session_scratch_e358/stage
out=experiments/E358_wiring-head-on-attached-ends/staged_diff.patch
: > "$out"
for f in evaluation/qends_eval.py evaluation/jends_eval.py training/grid/jwire_head.py; do
  b=$(basename "$f")
  diff -u --label "a/$f" --label "b/stage/$b" "$f" "$S/$b" >> "$out"
done
wc -l "$out"
grep -c '^@@' "$out"

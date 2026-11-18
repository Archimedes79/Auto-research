#!/usr/bin/env bash
# the three test files of the champion (CPU only, as the files say); output to temp/jr/jtoken_tests.log
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
: > temp/jr/jtoken_tests.log
for t in training/grid/test_jtoken.py training/grid/test_lane_tokcache.py evaluation/tests/test_jtoken_eval.py; do
  echo "=== $t $(date +%T)" >> temp/jr/jtoken_tests.log
  bash tools/env/_py14.sh $t >> temp/jr/jtoken_tests.log 2>&1
  echo "exit $? $(date +%T)" >> temp/jr/jtoken_tests.log
done
echo DONE >> temp/jr/jtoken_tests.log

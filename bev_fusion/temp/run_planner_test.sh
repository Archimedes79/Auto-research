#!/bin/bash
# scratch: does the planner's runner (runner/bev_runner.py) still pass its own test? (CPU, ~10 min), real exit code in the log
cd /mnt/c/Auto-research/bev_fusion || exit 1
OMP_NUM_THREADS=3 ~/miniconda3/envs/gcar/bin/python tests/test_runner.py > temp/planner_test.log 2>&1
echo "exit $?" >> temp/planner_test.log

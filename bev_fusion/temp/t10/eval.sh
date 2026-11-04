#!/bin/bash
# eval of t10_dual_s0 (cold VAL 1000), same call as lane4 for the t8 arms
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY -u evaluation/bev_suite.py runs/t10_dual_s0 --workers 4 eval --n 1000 > temp/t10/eval_t10_dual_s0.log 2>&1
echo "exit $?" >> temp/t10/eval_t10_dual_s0.log

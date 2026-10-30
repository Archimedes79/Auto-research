#!/usr/bin/env bash
# One-off: full-val evaluation of the baseline t1_r0 at low priority next to the running final training.
cd "$(dirname "$0")/../.." || exit 1
nice -n 15 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t1_r0 --workers 3 eval --prior 500 --images 8 > temp/train/eval_r0.log 2>&1
echo "eval_r0 exit $? $(date +%T)"

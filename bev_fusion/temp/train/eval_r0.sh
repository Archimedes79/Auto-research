#!/usr/bin/env bash
# One-off: full-val evaluation of the baseline t1_r0 at low priority next to the running final training.
cd "$(dirname "$0")/../.." || exit 1
nice -n 15 ~/miniconda3/envs/gcar/bin/python -u evaluation/evaluate_run.py runs/t1_r0 --prior 500 --images 8 --workers 3 --batch 4 > temp/train/eval_r0.log 2>&1
echo "eval_r0 exit $? $(date +%T)"

#!/bin/bash
# evaluation of t10_dual_long_s0 (the 6000-step run of card sections 3b / 3c): cold VAL 1000, the same call as for t10_dual_s0 / s1; the paired comparisons of card 3c follow by hand
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY -u evaluation/bev_suite.py runs/t10_dual_long_s0 --workers 4 eval --n 1000

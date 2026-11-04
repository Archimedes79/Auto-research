#!/bin/bash
# evaluation of t10_dual_s1 (the seed twin of the dual chain): cold VAL 1000, the same call as for t10_dual_s0 and the t8 arms
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY -u evaluation/bev_suite.py runs/t10_dual_s1 --workers 4 eval --n 1000

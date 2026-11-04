#!/bin/bash
# smoke test e of the D32 merge: the evaluation of runs/smoke_d32 on 20 val frames (the same suite call as the real runs, smaller n)
cd "$(dirname "$0")/../.." || exit 1
~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/smoke_d32 --workers 2 eval --n 20

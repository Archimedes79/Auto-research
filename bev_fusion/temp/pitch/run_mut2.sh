#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
nice -n 12 $PY evaluation/bev_response.py --n 50 --selftest 2>&1 | grep -E "OFF|selftest|Error|Traceback|rows off" | cut -c1-200
nice -n 12 $PY -u temp/pitch/mutants.py 24 AVE "TP threshold" AOE 2>&1 | cut -c1-200

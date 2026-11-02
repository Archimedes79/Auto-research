#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
rm -rf temp/pitch/modules_check temp/pitch/timing_check.json
nice -n 10 $PY -u evaluation/bev_suite.py runs/t16_small_s0 modules --frames 4 --threads 2 --out temp/pitch/modules_check > temp/pitch/modules_check.log 2>&1
echo "exit $?" >> temp/pitch/modules_check.log
nice -n 10 $PY -u evaluation/bev_suite.py runs/t16_small_s0 --device cpu timing --n 2 --warmup 1 --out temp/pitch/timing_check.json > temp/pitch/timing_check.log 2>&1
echo "exit $?" >> temp/pitch/timing_check.log

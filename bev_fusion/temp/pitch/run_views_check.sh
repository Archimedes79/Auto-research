#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
rm -rf temp/pitch/views_check
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t16_small_s0 --device cpu --workers 0 --out temp/pitch/views_check qual frame scene-0103/10 --view overview outputs dashboard > temp/pitch/views_check.log 2>&1
echo "exit $?" >> temp/pitch/views_check.log

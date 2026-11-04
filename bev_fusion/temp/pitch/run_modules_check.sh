#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t16_small_s0 modules --frames 4 --threads 2 --out temp/pitch/modules_check > temp/pitch/modules_check.log 2>&1
echo "exit $?" >> temp/pitch/modules_check.log

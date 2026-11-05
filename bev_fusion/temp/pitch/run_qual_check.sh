#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
rm -rf temp/pitch/qual_check
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t16_small_s0 --device cpu --workers 0 --out temp/pitch/qual_check qual debug scene-0103/10 > temp/pitch/qual_check.log 2>&1
echo "exit $?" >> temp/pitch/qual_check.log

#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python evaluation/bev_selftest.py > temp/pitch/selftest_adapter.log 2>&1
echo "exit $?" >> temp/pitch/selftest_adapter.log

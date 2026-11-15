#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 15 ~/miniconda3/envs/gcar/bin/python tests/test_radar_recursive.py > temp/t25_review/main_test.log 2>&1
echo "exit $?" >> temp/t25_review/main_test.log

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
P=~/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
$P tests/test_bev_free.py runs/t16_full_s0 > temp/cleanup/test_free4.log 2>&1
echo "exit $?" >> temp/cleanup/test_free4.log

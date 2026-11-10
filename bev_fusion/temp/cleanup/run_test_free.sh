#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
P=~/miniconda3/envs/gcar/bin/python
$P -m py_compile evaluation/bev_free.py tests/test_bev_free.py || exit 1
uptime
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
$P tests/test_bev_free.py runs/t20_base_s0 > temp/cleanup/test_free.log 2>&1
echo "exit $?" >> temp/cleanup/test_free.log

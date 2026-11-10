#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES=""
$HOME/miniconda3/envs/gcar/bin/python tests/test_frozen.py 2>&1 | tail -2
echo "---waiter:"
ps -eo pid,etime,args | grep -E 'night_c0' | grep -v grep | cut -c1-110

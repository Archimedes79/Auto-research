#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
$HOME/miniconda3/envs/gcar/bin/python temp/t22/mutate_moved.py > temp/t22/chain/mutate_moved.log 2>&1

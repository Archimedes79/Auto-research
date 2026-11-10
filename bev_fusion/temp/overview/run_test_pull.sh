#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nice -n 12 $HOME/miniconda3/envs/gcar/bin/python tests/test_lane_pull.py 2>&1 | tail -n 45

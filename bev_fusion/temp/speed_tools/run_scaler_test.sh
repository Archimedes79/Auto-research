#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python tests/test_grad_scaler.py 2>&1 | tail -${1:-30} | cut -c1-230

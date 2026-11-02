#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 12 ~/miniconda3/envs/gcar/bin/python -u temp/pitch/split_counts.py runs/t10_dual_s0 2>&1 | tail -16 | cut -c1-220

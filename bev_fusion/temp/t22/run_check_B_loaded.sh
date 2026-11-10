#!/bin/bash
# scratch: CPU check of what the evaluator's loader gives for t22_B_s0 (2 threads, low priority)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
nice -n 10 $HOME/miniconda3/envs/gcar/bin/python temp/t22/check_B_loaded.py 2>&1 | cut -c1-500 | tail -20

#!/bin/bash
# t26.base_eval43: the same base on ALL val frames of the 43 scenes where both references exist (cold), CPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
nice -n 15 ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py temp/t26/base_run --device cpu --workers 1 eval --scenes $(tr '\n' ' ' < temp/t26/scenes43.txt)

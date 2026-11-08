#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PYTHONPATH=../autoresearch nice -n 15 "$HOME/miniconda3/envs/gcar/bin/python" temp/t21/warm_check.py 2>&1 | grep -v -i warn | tail -12

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py training/configs/t21_pair4.json 2>&1 | tail -8 | cut -c1-240

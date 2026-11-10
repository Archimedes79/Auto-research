#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES=""
$HOME/miniconda3/envs/gcar/bin/python temp/t22/read_epe_C0.py 2>&1 | head -60

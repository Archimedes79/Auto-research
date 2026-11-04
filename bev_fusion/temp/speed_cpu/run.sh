#!/bin/bash
# usage: bash temp/speed_cpu/run.sh <script.py> [args]   (from anywhere; WSL, gcar python, nice, 2 threads)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec nice -n 15 ~/miniconda3/envs/gcar/bin/python "temp/speed_cpu/$@"

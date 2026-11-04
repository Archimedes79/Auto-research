#!/bin/bash
# usage: run_one_test.sh <test name without .py> [lines]   (from anywhere; CPU, nice, 2 threads)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)"
OMP_NUM_THREADS=2 nice -n 19 ~/miniconda3/envs/gcar/bin/python tests/$1.py 2>&1 | tail -${2:-40} | cut -c1-260

#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py runs/t10_dual_s0 --device cpu --workers 0 list 2>&1 | head -60

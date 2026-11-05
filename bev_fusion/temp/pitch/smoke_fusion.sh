#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
rm -rf temp/pitch/smoke_fusion_out
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/t10_dual_s0 --device cpu --workers 0 --curves --fusion --out temp/pitch/smoke_fusion_out eval --n 6 > temp/pitch/smoke_fusion.log 2>&1

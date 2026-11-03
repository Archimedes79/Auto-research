#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_mutants.py 24 neutral "outside the fusion" hidden > temp/pitch/mut_fusion.log 2>&1

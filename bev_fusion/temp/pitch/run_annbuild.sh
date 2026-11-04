#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 12 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_annotation_pts.py build > temp/pitch/annbuild.log 2>&1
cat temp/pitch/annbuild.log | tail -3
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 12 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_annotation_pts.py check 2>&1 | tail -3

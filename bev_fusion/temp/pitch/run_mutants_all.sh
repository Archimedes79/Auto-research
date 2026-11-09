#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 12 ~/miniconda3/envs/gcar/bin/python -u temp/pitch/mutants.py 24 > temp/pitch/mutants_all.log 2>&1
tail -16 temp/pitch/mutants_all.log | cut -c1-200

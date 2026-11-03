#!/bin/bash
# scratch: one overview video of the baseline on the CPU (2 threads, low priority) while the GPU is full: how long does it take?
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/home/ml/miniconda3/envs/gcar/bin/python
date +%T
time nice -n 15 $PY evaluation/bev_suite.py runs/t16_full_s0 --device cpu --workers 1 --out runs/t16_full_s0/eval/showcase qual videos 1 --view overview --seed 11 2>&1 | tail -8 | cut -c1-240
date +%T

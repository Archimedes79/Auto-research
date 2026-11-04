#!/bin/bash
# scratch: 8 random val scenes of the baseline as overview videos on the CPU (2 threads, low priority) while the GPU holds the t18.fix training.
# seed 21 picks the same scenes for every run, so the other models get the same scenes later (GPU, in their evaluation chain)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/home/ml/miniconda3/envs/gcar/bin/python
date +%T
nice -n 15 $PY evaluation/bev_suite.py runs/t16_full_s0 --device cpu --workers 1 --out runs/t16_full_s0/eval/showcase qual videos 8 --view overview --seed 21 2>&1 | tail -12 | cut -c1-200
date +%T

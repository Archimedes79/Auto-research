#!/bin/bash
# scratch: paired scene bootstrap of the t18 bundle arm against its control t17_data (the card's rule) and against the baseline t16_full_s0, VAL 1000 frames, GT v8
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/home/ml/miniconda3/envs/gcar/bin/python
nice -n 15 $PY evaluation/bev_suite.py compare runs/t17_data runs/t18_fix --curves --fusion --velocity --n 1000 --boot 1000 > temp/t16/compare_t18.txt 2>&1
echo "rc=$? t17 vs t18"
nice -n 15 $PY evaluation/bev_suite.py compare runs/t16_full_s0 runs/t18_fix --curves --fusion --velocity --n 1000 --boot 1000 > temp/t16/compare_t18_vs_base.txt 2>&1
echo "rc=$? base vs t18"

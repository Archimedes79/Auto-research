#!/bin/bash
# scratch: four overview videos of the baseline on the lane-marking scenes of the gallery (best / worst / median lane-divider IoU of different scenes, lane_frames_t16_full_s0.txt),
# on the CPU (2 threads, low priority); Mirko asked for more different result videos at 00:15 while t18.fix trains (the training slows by about 25 % during this)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/home/ml/miniconda3/envs/gcar/bin/python
date +%T
nice -n 15 $PY evaluation/bev_suite.py runs/t16_full_s0 --device cpu --workers 1 --out runs/t16_full_s0/eval/showcase qual videos 4 --view overview --scenes scene-0016 scene-0017 scene-0273 scene-0099 2>&1 | tail -6 | cut -c1-200
date +%T

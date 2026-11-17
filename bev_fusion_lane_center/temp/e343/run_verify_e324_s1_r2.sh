#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 bash tools/env/_py14.sh jtest_eval.py verify --run e324_s_s1 > temp/e343/verify_e324_s1_r2.log 2>&1
echo "exit $?" >> temp/e343/verify_e324_s1_r2.log

#!/bin/bash
# E343 code step: regression of E312's path (VAL, stored dumps, CPU, one thread): verify --seed 0, then 1
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for S in 0 1; do
  nice -n 10 bash tools/env/_py14.sh jtest_eval.py verify --seed $S > temp/e343/verify_e312_s$S.log 2>&1
  echo "exit $?" >> temp/e343/verify_e312_s$S.log
done

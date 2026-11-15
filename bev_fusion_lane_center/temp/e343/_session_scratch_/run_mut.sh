#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 bash tools/env/_py14.sh temp/e343/_session_scratch_/mutate.py --part $1/2 > temp/e343/_session_scratch_/mutate_r2_$1.log 2>&1

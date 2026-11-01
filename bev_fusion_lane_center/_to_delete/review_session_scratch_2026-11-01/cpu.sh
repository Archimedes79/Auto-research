#!/bin/bash
# run a project script on the CPU with 2 threads, report the peak memory: bash cpu.sh <script> args...
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
/usr/bin/time -f "peak RSS %M KB, %e s" bash tools/env/_py14.sh "$@"

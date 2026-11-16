#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh e276_build_report.py 2>&1 | tail -15

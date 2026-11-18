#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh /mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_prof/prof_tokstep.py > temp/_session_scratch_prof/prof.log 2>&1
echo "exit $?" >> temp/_session_scratch_prof/prof.log

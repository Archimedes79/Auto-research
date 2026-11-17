#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=4
for a in z m; do bash tools/env/_py14.sh temp/_session_scratch_d113/work3/mem_probe2.py $a 2>&1 | tail -3; done

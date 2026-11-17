#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
bash tools/env/_py14.sh test_jmap_train.py selftests > temp/_session_scratch_d113/work3/tt_out.txt 2>&1
echo done >> temp/_session_scratch_d113/work3/tt_out.txt

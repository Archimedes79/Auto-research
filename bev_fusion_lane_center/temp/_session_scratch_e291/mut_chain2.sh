#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 temp/e291/mutate.py s01_term_not_added > temp/_session_scratch_e291/mut_s01.log 2>&1
echo "rc $?" >> temp/_session_scratch_e291/mut_s01.log

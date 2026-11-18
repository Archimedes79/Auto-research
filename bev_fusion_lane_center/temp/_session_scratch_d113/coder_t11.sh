#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/_session_scratch_d113
bash tools/env/_py14.sh test_lane_past.py > $L/coder_u_past.log 2>&1
bash tools/env/_py14.sh test_jpast_train.py > $L/coder_u_train.log 2>&1
echo finished > $L/coder_t11.done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/_session_scratch_d113
bash tools/env/_py14.sh test_jpast.py > $L/coder_w_jpast.log 2>&1
bash tools/env/_py14.sh test_jpast_train.py > $L/coder_w_train.log 2>&1
echo finished > $L/coder_t13.done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
bash tools/env/_py14.sh test_jpast.py > temp/_session_scratch_d113/coder_t_jpast.log 2>&1
bash tools/env/_py14.sh test_jpast_train.py > temp/_session_scratch_d113/coder_t_train.log 2>&1
bash tools/env/_py14.sh test_lane_past.py > temp/_session_scratch_d113/coder_t_past.log 2>&1
echo finished > temp/_session_scratch_d113/coder_t4.done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
bash tools/env/_py14.sh lane_past_data.py check > temp/_session_scratch_d113/coder_final_check.log 2>&1
bash tools/env/_py14.sh lane_past_data.py fixture --split val --n-pairs 200 > temp/_session_scratch_d113/coder_final_fixture.log 2>&1
echo finished > temp/_session_scratch_d113/coder_t7.done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
L=temp/_session_scratch_d113
bash tools/env/_py14.sh lane_past_data.py selftest > $L/coder_s1.log 2>&1
bash tools/env/_py14.sh lane_past_data.py loader-fixture --split val --n-pairs 40 > $L/coder_s2.log 2>&1
bash tools/env/_py14.sh test_jpast_train.py one /mnt/c/Auto-research/bev_fusion_lane_center/training/train/jpast_train.py /mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jpast_eval.py > $L/coder_s3.log 2>&1
bash tools/env/_py14.sh test_jpast_train.py real > $L/coder_s4.log 2>&1
echo finished > $L/coder_t9.done

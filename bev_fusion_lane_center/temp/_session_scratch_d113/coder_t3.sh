#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
bash tools/env/_py14.sh test_jpast_train.py one /mnt/c/Auto-research/bev_fusion_lane_center/training/train/jpast_train.py /mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jpast_eval.py

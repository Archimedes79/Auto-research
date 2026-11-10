#!/bin/bash
# scratch: symbol scan of my image-encoder files (output to temp/cleanup/t22_symbol_scan.txt)
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/cleanup/t22_symbol_scan.py \
  bev_fusion/training/student_encoder.py bev_fusion/training/distill_student.py bev_fusion/training/camera_encoder.py bev_fusion/training/bevfusion_camera.py \
  bev_fusion/training/lift_pd.py bev_fusion/training/lane_branch.py bev_fusion/training/cache_bevfusion_features.py bev_fusion/training/feature_cache.py \
  bev_fusion/training/interface.py bev_fusion/training/startable.py > bev_fusion/temp/cleanup/t22_symbol_scan.txt 2>&1
echo "scan done $(date +%H:%M:%S) exit $?"

#!/usr/bin/env bash
# t21 scratch: the gates that touch the camera temporal path, one line each
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in training/temporal_fusion.py training/motion_block.py training/timed_head.py tests/test_bundle.py tests/test_layers.py tests/test_lane_branch_dual.py tests/test_camera_features.py tests/test_motion_joint.py tests/test_slots.py training/interface.py; do
  s=$(date +%s)
  if out=$(nice -n 15 $PY $t 2>&1); then r=PASS; else r=FAIL; fi
  echo "$r $t $(( $(date +%s) - s )) s | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-150)"
done

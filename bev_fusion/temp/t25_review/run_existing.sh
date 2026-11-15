#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
L=temp/t25_review/existing.log
: > $L
for t in training/interface.py training/precompute.py tests/test_bundle.py tests/test_radar_memory_ghost.py tests/test_motion_joint.py tests/test_stream_temporal.py tests/test_slots.py training/augment.py tests/test_clip_split.py tests/test_launch_saving.py tests/test_camera_features.py tests/test_train_guards.py tests/test_student_chain.py tests/test_lane_branch_dual.py tests/test_speed_flags.py tests/test_host_batch.py; do
  s=$(date +%s)
  out=$(nice -n 15 $PY $t 2>&1); rc=$?
  e=$(date +%s)
  echo "== $t rc=$rc $((e-s))s | $(echo "$out" | grep -v -i warn | tail -2 | tr '\n' ' ' | cut -c1-260)" >> $L
done
echo finished >> $L

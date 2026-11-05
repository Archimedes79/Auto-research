#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
BOOT=100 $PY temp/t4/compare_front.py experiments/t10_dual_features/results_x50.md 50 \
  t8_ctl_s0=runs/t8_ctl_s0/eval/scene_records_n1000.pkl \
  t10_dual_s0=runs/t10_dual_s0/eval/scene_records_n1000.pkl \
  t8_bevfeat_s0=runs/t8_bevfeat_s0/eval/scene_records.pkl \
  t8_bevfeat_s1=runs/t8_bevfeat_s1/eval/scene_records.pkl > temp/t10/compare.log 2>&1
echo "exit $?" >> temp/t10/compare.log

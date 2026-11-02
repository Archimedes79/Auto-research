#!/bin/bash
# front-sector comparison with both seeds of the control and of the dual chain (t8_ctl_s0 is the reference of the bootstrap)
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
OMP_NUM_THREADS=2 BOOT=100 nice -n 10 $PY temp/t4/compare_front.py experiments/t10_dual_features/results_x50_2seeds.md 50 \
  t8_ctl_s0=runs/t8_ctl_s0/eval/scene_records_n1000.pkl \
  t8_ctl_s1=runs/t8_ctl_s1/eval/scene_records_n1000.pkl \
  t10_dual_s0=runs/t10_dual_s0/eval/scene_records_n1000.pkl \
  t10_dual_s1=runs/t10_dual_s1/eval/scene_records_n1000.pkl \
  t8_bevfeat_s0=runs/t8_bevfeat_s0/eval/scene_records.pkl \
  t8_bevfeat_s1=runs/t8_bevfeat_s1/eval/scene_records.pkl > temp/t10/compare2.log 2>&1
echo "exit $?" >> temp/t10/compare2.log

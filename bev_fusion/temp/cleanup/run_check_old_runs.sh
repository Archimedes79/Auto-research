#!/bin/bash
# scratch: load the stored t22 runs through the evaluation's own loader, one process each (a stand-in run keeps its weights in the run it was made from)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=~/miniconda3/envs/gcar/bin/python
for spec in "t22_B2_s0:" "t22_B_s0:" "t22_B_all:runs/t22_B_s0" "t22_standin_all_pair4:runs/t21_pair4_s0" "smoke_t22_all_cpu:runs/t21_pair4_s0" "t22_standin_pair4:runs/t21_pair4_s0" "t22_twin_pair4:runs/t21_pair4_s0"; do
  r=${spec%%:*}; c=${spec#*:}
  $PY temp/cleanup/check_old_runs.py runs/$r $c 2>&1 | grep -E "^(LOADS|FAILS|CONTROLS)" | cut -c1-400
done
echo "done $(date +%H:%M:%S)"

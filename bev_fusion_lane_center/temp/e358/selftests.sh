#!/bin/bash
# E358 scratch: the four selftests of the files of the step (CPU, nice, 2 threads), one after the other:  wsl bash temp/e358/selftests.sh [names]
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
names="${@:-jwire_data jwire_train jwire_check jwire_verdict}"
for t in $names; do
  s=$(date +%s)
  nice -n 10 bash tools/env/_py14.sh $t.py selftest 2>&1 | tail -4
  echo "== $t: exit ${PIPESTATUS[0]}, $(( $(date +%s) - s )) s"
done

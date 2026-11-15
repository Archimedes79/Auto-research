#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in test_verdict277.py test_verdict275.py test_tier_metrics.py; do
  echo "== $t"
  nice -n 10 bash tools/env/_py14.sh $t 2>&1 | tail -6
done

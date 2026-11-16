#!/bin/bash
# scratch: the selftests of the modules E277 changed (CPU), default path must be unchanged
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in qnodes.py qgraph.py test_tier_loss.py test_soft_nms.py; do
  echo "== $t"
  nice -n 10 bash tools/env/_py14.sh $t 2>&1 | tail -4
done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
for t in sparse_lanes/selftest.py sparse_lanes/topo_metrics.py test_evaluate_guard.py test_p18_joint_guard.py test_lane_prototype.py test_eval_parts.py; do
  echo "== $t"; CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh $t 2>&1 | tail -4; echo "rc=${PIPESTATUS[0]}"
done

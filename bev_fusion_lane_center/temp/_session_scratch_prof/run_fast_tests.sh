#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for t in test_jtoken_fast test_lane_ptok_gpu test_build_tokcache_root; do
  CUDA_VISIBLE_DEVICES= nice -n 10 bash tools/env/_py14.sh training/grid/$t.py 2>&1 | tail -5
done

#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in test_qgraph_score_glue.py test_verdict274.py test_qgraph_readout.py grid/test_tiers_sparse.py; do
  echo "== $t"
  nice -n 10 bash tools/env/_py14.sh $t 2>&1 | tail -3
done

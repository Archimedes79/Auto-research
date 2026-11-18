#!/bin/bash
# reviewer: the tests that touch the lean scoring change (measures.topo_counts marbles flag, qgraph_eval lean, jtoken_eval.score_preds)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for t in test_lane_task.py test_tier_metrics.py test_qgraph_score_glue.py test_jtoken_eval.py test_token_readout.py; do
  s=$(date +%s)
  out=$(nice -n 10 bash tools/env/_py14.sh $t 2>&1); rc=$?
  echo "$(date +%H:%M:%S) rc=$rc $t ($(( $(date +%s) - s )) s): $(echo "$out" | tail -1 | cut -c1-150)"
done
echo "all done"

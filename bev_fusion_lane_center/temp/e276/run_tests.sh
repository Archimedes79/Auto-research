#!/bin/bash
# scratch: the lane scorer / target tests on CPU (2 threads)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for t in test_qgraph_readout.py test_verdict275.py test_qgraph_score_glue.py test_tier_metrics.py test_tiers_sparse.py test_tier_loss.py test_qgraph_eval.py test_qtargets.py; do
  echo "== $t"
  bash tools/env/_py14.sh $t 2>&1 | tail -3
done
ls /mnt/c/Auto-research/data/autogt/lane/c96c18fb1f | wc -l
date +%H:%M

#!/bin/bash
# E276: wait for the full l4 store, check it (read-only), make READY and start the scheduler. Log temp/e276/go.log.
cd "$(dirname "$0")/../.."
S=/mnt/c/Auto-research/data/autogt/lane/0702a065c6
L3=/mnt/c/Auto-research/data/autogt/lane/3576fa1f7e
{
  until [ -f /mnt/c/Auto-research/temp/gt_l4/GEN_DONE ]; do sleep 20; done
  echo "GEN_DONE $(date)"
  n=$(ls $S | grep -c '\.pkl\.gz$')
  echo "files in the store: $n (want 34149)"
  [ "$n" -eq 34149 ] || { echo "STORE INCOMPLETE: no READY"; exit 1; }
  grep -q '"complete": true' $S/index.json && echo "index complete" || echo "index not complete (the GT chat refreshes it): continuing on the file count"
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
  echo "== geometry check l3 vs l4 $(date)"
  nice -n 12 bash tools/env/_py14.sh check_geometry.py $L3 $S 500 || { echo "GEOMETRY DIFFERS: no READY"; exit 2; }
  echo "== tier statistics of the full store $(date)"
  nice -n 12 bash tools/env/_py14.sh exp_tier_stats.py $S 3000 2>&1 | tail -22
  echo "== quick tests $(date)"
  for t in test_tier_loss.py test_tier_metrics.py test_tiers_sparse.py test_qgraph_score_glue.py; do
    nice -n 12 bash tools/env/_py14.sh $t 2>&1 | tail -1 | cut -c1-110
  done
  touch temp/e276/READY
  echo "READY $(date)"
  nohup bash temp/e276/launch_chain.sh > temp/e276/chain.out 2>&1 < /dev/null &
  sleep 5
  pgrep -af "[l]aunch_chain" | head -2 | cut -c1-90
  echo "scheduler started $(date)"
} > temp/e276/go.log 2>&1

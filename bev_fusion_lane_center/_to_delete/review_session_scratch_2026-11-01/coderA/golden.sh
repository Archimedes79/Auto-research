#!/bin/bash
# throw-away: the CPU part of run_all_tests.sh that touches the bootstrap / metric imports (no GPU)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for t in "evaluate.py --golden check --cached-only"; do
  echo "=== $t"; start=$(date +%s)
  bash tools/env/_py14.sh $t 2>&1 | tail -4; echo "rc=${PIPESTATUS[0]} ($(( $(date +%s) - start )) s)"
done

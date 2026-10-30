#!/usr/bin/env bash
# One-off: evaluate 800 TRAIN samples of a run with the val code path (overfitting check) -> runs/<run>/eval/*_train.*
# Usage: bash temp/train/run_trainsub.sh t1_r4_soa [t1_r0 ...]
cd "$(dirname "$0")/../.." || exit 1
for r in "$@"; do
  echo "== $r train subset $(date +%T)"
  ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py runs/$r --workers 4 --split train eval --n 800 > temp/train/trainsub_$r.log 2>&1
  tail -n 2 temp/train/trainsub_$r.log
done
echo "trainsub finished $(date +%T)"

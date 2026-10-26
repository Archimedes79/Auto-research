#!/usr/bin/env bash
# One-off: 30-step speed runs of the t1 chain, micro-batch 2 x 8 and 4 x 4 (seconds per optimiser step, GPU memory).
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for v in b2 b4; do
  echo "=== speed_$v $(date +%T)"
  $PY -u training/train.py temp/train/speed_$v.json 2>&1 | tail -n 12
done
echo "speed runs done $(date +%T)"

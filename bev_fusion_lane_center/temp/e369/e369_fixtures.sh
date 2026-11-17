#!/bin/bash
# E369 fixture (throw-away wrapper): TRAIN sample (every 3, 291 frames) for both seeds first, then VAL; CPU, niced, 2 threads
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for split in train val; do
  for s in 0 1; do
    extra=""; [ "$split" = train ] && extra="--every 3"
    echo "=== fixture seed $s $split $(date +%H:%M:%S)"
    nice -n 10 bash tools/env/_py14.sh evaluation/jtoken_eval.py fixture --seed $s --split $split $extra 2>&1 | grep -v "frames, [0-9]* s$"
  done
done
echo "=== done $(date +%H:%M:%S)"

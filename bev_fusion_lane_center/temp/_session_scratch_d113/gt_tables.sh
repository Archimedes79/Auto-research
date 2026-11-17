#!/bin/bash
# D113: the GT node tables of VAL and TRAIN on the 2 m lattice (B's own grid) into the E357 table cache, one CPU process, 2 threads
cd "$(dirname "$0")/../.." || exit 1
for s in val train; do
  bash tools/env/_py14.sh training/data/lane_tok_data.py build-cache --split $s --lattice 2.0 --threads 2 || exit 1
done
echo "gt tables done $(date +%H:%M:%S)"

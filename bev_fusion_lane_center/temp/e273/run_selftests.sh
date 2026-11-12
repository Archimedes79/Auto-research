#!/bin/bash
# one-off: the E273 module self-tests on the CPU (1 thread)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
for f in qtargets qformat map_sim qraster pyramid qnodes qlinks sections qassign qgraph; do
  echo "== $f"; nice -n 12 bash tools/env/_py14.sh grid/$f.py 2>&1 | tail -1 | cut -c1-160
done

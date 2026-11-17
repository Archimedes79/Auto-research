#!/bin/bash
# seed 1 of the error analysis (err_taxonomy, tier_view, piece_found), one after the other, 1 thread each (shared box)
cd "$(dirname "$0")/../.." || exit 1
for s in 1; do
  for sc in err_taxonomy tier_view piece_found; do
    echo "=== $sc seed $s"
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/$sc.py $s
  done
done
echo "batch done"

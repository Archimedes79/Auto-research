#!/bin/bash
# fork C scratch: verify the jtest_eval scorer on VAL against the stored E312 cards (stored dumps), both seeds
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
rm -f temp/e321/verify.done
for S in 0 1; do
  /usr/bin/time -v nice -n 10 bash tools/env/_py14.sh evaluation/jtest_eval.py verify --seed $S --boot 100 > temp/e321/verify_s$S.log 2>&1
done
echo done > temp/e321/verify.done

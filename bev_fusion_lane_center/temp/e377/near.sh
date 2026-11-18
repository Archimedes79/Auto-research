#!/bin/bash
# E377 v2 card measurement (scratch): the near-an-assigned-token label on TRAIN (both caches) for the champion and the weight-1.0 heads, VAL for the champion
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for x in "e373_cb_s0 0 train 3" "e373_cb_s1 1 train 3" "e371_cf_s0 0 train 3" "e367_b5_s0 0 train 3" "e373_cb_s0 0 val 1"; do
  set -- $x
  nice -n 10 bash tools/env/_py14.sh e377_near.py $1 $2 $3 $4 > temp/e377/near_$1_$3.out 2>&1
  echo "$x exit $?"
done

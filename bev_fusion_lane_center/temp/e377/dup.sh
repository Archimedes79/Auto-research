#!/bin/bash
# E377 card measurement (scratch): duplicates on the TRAIN sample (every 3) of the none-weight family on the same seeds (E371 cf = 1.0, E373 cb = 0.25) and E367 b5 (0.5), then VAL of the champion heads
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for x in "e373_cb_s0 0 train 3" "e371_cf_s0 0 train 3" "e367_b5_s0 0 train 3" "e373_cb_s1 1 train 3" "e371_cf_s1 1 train 3" "e373_cb_s0 0 val 1" "e373_cb_s1 1 val 1" "e371_cf_s0 0 val 1"; do
  set -- $x
  nice -n 10 bash tools/env/_py14.sh e377_dup.py $1 $2 $3 $4 > temp/e377/dup_$1_$3.out 2>&1
  echo "$x exit $?"
done

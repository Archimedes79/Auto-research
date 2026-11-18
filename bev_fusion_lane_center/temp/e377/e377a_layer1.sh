#!/bin/bash
# E377 layer 1 (scratch): the anatomy's near_none section of the scratch copy of jtoken_eval.py on e373_cb_s0, TRAIN every 3 then VAL
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for x in "train 3" "val 1"; do
  set -- $x
  nice -n 10 bash tools/env/_py14.sh e377a_layer1.py $1 $2 temp/e377/e377a_session_scratch_ > temp/e377/e377a_layer1_$1.out 2>&1
  echo "$x exit $?" >> temp/e377/e377a_layer1_$1.out
done

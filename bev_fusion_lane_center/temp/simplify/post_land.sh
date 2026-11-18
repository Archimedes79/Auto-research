#!/bin/bash
# after the landing of U1-U5 (2ade9bd41d): the token tests in the LIVE tree, the S4 gate argument parse, the repro of the stored cards
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="nice -n 10 bash tools/env/_py14.sh"
for s in training/grid/test_jtoken.py "training/grid/test_jlive_train.py" evaluation/tests/test_jtoken_eval.py "tools/jarm.py selftest"; do
  n=$(basename ${s%% *} .py)
  echo "=== $s"
  $P $s > temp/simplify/post_$n.out 2>&1
  rc=$?
  grep -iE " ok$|ok$|Error|FAIL|skip" temp/simplify/post_$n.out | tail -n 4 | cut -c1-200
  echo "rc $rc"
done
echo "=== repro"
bash temp/simplify/repro.sh post > temp/simplify/post_repro.out 2>&1
echo "rc $?"; tail -n 3 temp/simplify/post_repro.out
echo "post_land done"

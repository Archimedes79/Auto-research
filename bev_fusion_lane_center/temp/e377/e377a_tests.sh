#!/bin/bash
# E377 (scratch): the tests of evaluation/jtoken_eval.py and of the modules that import it, CPU only, after the move
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
for x in "test_jtoken_eval.py" "test_jpast_train.py selftests" "test_jmap_train.py selftests" "test_jtoken.py"; do
  set -- $x
  nice -n 10 bash tools/env/_py14.sh $x > temp/e377/e377a_tests_${1%.py}.out 2>&1
  echo "$x exit $?" | tee -a temp/e377/e377a_tests_${1%.py}.out
done

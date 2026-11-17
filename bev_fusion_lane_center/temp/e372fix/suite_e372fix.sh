#!/bin/bash
# throw-away: the two test files of the step, in full, on the real files (2 threads, nice)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for t in training/grid/test_jtoken.py evaluation/tests/test_jtoken_eval.py; do
  nice -n 10 bash tools/env/_py14.sh "$t" > "temp/e372fix/$(basename "$t" .py).log" 2>&1
  echo "$t exit $?"
done

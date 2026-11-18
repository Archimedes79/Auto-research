#!/bin/bash
# the moved-in test file from its live place (CPU, GPU hidden); usage: e379e_tests_live.sh <out file> [test names]
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
out="$1"; shift
nice -n 10 bash tools/env/_py14.sh test_jtoken_eval.py "$@" > "$out" 2>&1
echo "exit $?" >> "$out"
nice -n 10 bash tools/env/_py14.sh jtoken_eval.py ensemble -h >> "$out" 2>&1

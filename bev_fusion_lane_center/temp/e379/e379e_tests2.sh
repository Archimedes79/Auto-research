#!/bin/bash
# the test file of jtoken_eval.py on the second dev copies (CPU, GPU hidden); usage: e379e_tests2.sh <out file> [test names]
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
out="$1"; shift
nice -n 10 bash tools/env/_py14.sh temp/e379/e379e_session_scratch_dev2/evaluation/tests/test_jtoken_eval.py "$@" > "$out" 2>&1
echo "exit $?" >> "$out"

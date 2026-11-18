#!/bin/bash
# U2 scratch: the tests of the worktree's edited modules against the live data (u2_run_with.py); the subprocess-based `selftests` modes as their subcommands; the speed-path test last (expected to fail: it uses the removed prior)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
W="nice -n 10 bash tools/env/_py14.sh u2_run_with.py"
R=$(pwd)
run() { echo "=== $*"; $W "$@" > temp/simplify/u2_last.out 2>&1; rc=$?; tail -n 3 temp/simplify/u2_last.out | cut -c1-600; echo "rc $rc"; [ $rc -eq 0 ] || cp temp/simplify/u2_last.out "temp/simplify/u2_fail_$(basename $1 .py)_$2.out"; }
run training/grid/test_jtoken.py
run training/grid/test_jlive_train.py selftests
run training/grid/test_jmap_train.py one "$R/training/train/jmap_train.py" "$R/evaluation/jmap_eval.py"
run training/grid/test_jmap_train.py real
run training/grid/test_jpast_train.py one "$R/training/train/jpast_train.py" "$R/evaluation/jpast_eval.py"
run training/grid/test_jpast_train.py real
run tools/jarm.py selftest
run evaluation/tests/test_jtoken_eval.py
run temp/simplify/repro_score.py u2
run training/grid/test_jtoken_fast.py
echo "u2_tests done"

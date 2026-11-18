#!/bin/bash
# U3-U5 scratch: the checks of one unit against the worktree's modules (u2_run_with.py: the edited modules by import finder, the live data); $1 = the unit tag (the repro's run folder runs_<tag> must not exist)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
W="nice -n 10 bash tools/env/_py14.sh u2_run_with.py"
T=$1
run() { echo "=== $*"; $W "$@" > temp/simplify/${T}_last.out 2>&1; rc=$?; grep -v "^u2_run_with" temp/simplify/${T}_last.out | tail -n 3 | cut -c1-500; echo "rc $rc"; [ $rc -eq 0 ] || cp temp/simplify/${T}_last.out "temp/simplify/${T}_fail_$(basename $1 .py).out"; }
run temp/simplify/u_gate_parse.py
run training/grid/test_jtoken.py
run training/grid/test_jlive_train.py selftests
run tools/jarm.py selftest
run evaluation/tests/test_jtoken_eval.py
run temp/simplify/repro_score.py $T
echo "${T}_tests done"

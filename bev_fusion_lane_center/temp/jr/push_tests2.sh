#!/bin/bash
# CPU-only selftests of the code added / changed since the last push (2026-11-15 evening)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="nice -n 19 bash tools/env/_py14.sh"
fail=0
run() { name="$1"; shift; out=$($P "$@" 2>&1); rc=$?; echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL)  $name | $(echo "$out" | tail -1 | cut -c1-120)"; [ $rc -eq 0 ] || fail=1; }
run "lanegraph tests (full)" training/lanegraph/tests/run_lg_tests.py
run "jtest_eval tests" evaluation/tests/test_jtest_eval.py
run "jnarrow tests" training/grid/test_jnarrow.py
run "jrule_removal selftest" evaluation/jrule_removal.py selftest
run "jgraph selftest" evaluation/jgraph.py selftest
run "jtiming selftest" evaluation/jtiming.py selftest
run "jdiag selftest" evaluation/jdiag.py selftest
run "jwire_check selftest" evaluation/jwire_check.py selftest
run "jwire_verdict selftest" evaluation/jwire_verdict.py selftest
run "jprior selftest" training/grid/jprior.py
echo "FAIL=$fail"

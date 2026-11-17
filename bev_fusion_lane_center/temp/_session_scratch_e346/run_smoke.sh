#!/bin/bash
# the final smoke of E346 (31 frames) + the read-only tests of the frozen files, one at a time, nice, 2 threads; log: temp/_session_scratch_e346/smoke.log
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=temp/_session_scratch_e346/smoke.log
: > $L
run() { echo "=== $* $(date +%T)" >> $L; nice -n 10 bash tools/env/_py14.sh "$@" >> $L 2>&1; echo "--- exit $? $(date +%T)" >> $L; }
run e346_arm.py selftest
for a in g0 g1 l2 l1; do run e346_arm.py $a --max 31 --skip-selftest; done
run e346_arm.py report --max 31
run qtargets.py
run qgraph_eval.py
run test_qgraph_eval.py
echo "ALL DONE $(date +%T)" >> $L

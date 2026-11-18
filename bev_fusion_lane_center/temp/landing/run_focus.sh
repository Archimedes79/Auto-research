#!/bin/bash
# the focused tests of the landing, three at a time (1 thread each, nice 10): perf patch (tokcache, live), the landed files (jtiming, lane_past)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { n=$1; shift; ( start=$(date +%s); nice -n 10 bash tools/env/_py14.sh "$@" > temp/landing/t/$n.out 2>&1; echo "rc=$? $(( $(date +%s) - start )) s" >> temp/landing/t/$n.out ) & }
run tokcache grid/test_lane_tokcache.py
run live grid/test_lane_live.py
run past grid/test_lane_past.py
wait
run jtiming jtiming.py selftest
run structure_dummy -c "print(1)"
wait
echo done > temp/landing/t/ALL_DONE

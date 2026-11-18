#!/bin/bash
# System test, stream B (2026-11-17 night, Mirko's order 21:45 'teste das System'): the token-champion tests that tools/maintenance/run_all_tests.sh does not list (19 files). Three at a time, niced, 2 threads each; one result line per file in temp/e383_tests/results.txt, the last 30 lines of each output in temp/e383_tests/logs/.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 BFLC_SIG=1.0
D=temp/e383_tests
run1() {
  f=$1; t0=$(date +%s)
  out=$(timeout 3600 nice -n 10 bash tools/env/_py14.sh "$f" 2>&1); rc=$?
  echo "$(date +%H:%M:%S) $([ $rc -eq 0 ] && echo PASS || echo FAIL) rc=$rc $f $(( $(date +%s) - t0 ))s | $(echo "$out" | tail -1 | cut -c1-150)" >> temp/e383_tests/results.txt
  echo "$out" | tail -30 > "temp/e383_tests/logs/$f.log"
}
export -f run1
printf '%s\n' test_jtoken_eval.py test_jtoken.py test_jmap_train.py test_jpast_train.py test_jlive_train.py test_lane_live.py test_lane_map.py test_jtok.py test_jtok2.py test_jrecur.py test_lane_past.py test_lane_tokcache.py test_jpast.py test_jmap.py test_build_tokcache_root.py test_lane_suite.py test_jtoken_fast.py test_lane_ptok_gpu.py | xargs -P 3 -I{} bash -c 'run1 {}'
echo "$(date +%H:%M:%S) stream B done" >> temp/e383_tests/results.txt

#!/bin/bash
# The LEGACY token tests that `run_all_tests.sh` does not list on purpose (since 2026-11-18 its last section lists the tests of the live champion path): the closed S2 / S3 blocks (E368 past, E374 map: both TIE by their written rules), the E354 recursive state, and the lane-suite selftest.
# Three at a time, niced, 2 threads each; one PASS / FAIL line per file, the last 30 lines of every output in the log folder. About 80 minutes on a shared box (test_jpast_train ~50 min under load, test_jmap_train ~44, test_lane_map ~37, test_lane_past ~27, test_jpast / test_jmap ~15 / 12,
# test_lane_suite ~9, test_jrecur ~2; CPU only, no GPU job needed). Last run 2026-11-17 / 18: all 8 PASS.
# usage (from the problem folder): wsl bash tools/maintenance/run_token_tests.sh [LOGDIR]      LOGDIR default temp/token_tests (scratch, ignored by git); exit 1 when any test failed.
cd "$(dirname "$0")/../.."
LOG=${1:-temp/token_tests}
mkdir -p "$LOG/logs"
: > "$LOG/results.txt"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 BFLC_SIG=1.0 LOG
run1() {
  f=$1; t0=$(date +%s)
  out=$(timeout 3600 nice -n 10 bash tools/env/_py14.sh "$f" 2>&1); rc=$?
  echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL)  $f  $(( $(date +%s) - t0 )) s | $(echo "$out" | tail -1 | cut -c1-150)" >> "$LOG/results.txt"
  echo "$out" | tail -30 > "$LOG/logs/$f.log"
}
export -f run1
printf '%s\n' test_jpast_train.py test_jmap_train.py test_lane_map.py test_lane_past.py test_jpast.py test_jmap.py test_lane_suite.py test_jrecur.py | xargs -P 3 -I{} bash -c 'run1 {}'
cat "$LOG/results.txt"
! grep -q "^FAIL" "$LOG/results.txt"

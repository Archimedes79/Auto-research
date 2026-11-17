#!/bin/bash
# The tests of the token champion (E357 - E379: the heads, the caches, the S2 / S3 / S4 blocks, the read-out and timing tools), which `run_all_tests.sh` does not list. Three at a time, niced, 2 threads each; one PASS / FAIL line per file, the last 30 lines of every output in the log folder.
# About 100 minutes on a shared box (the slow ones: test_jpast_train, test_jlive_train, test_jmap_train ~50 min each under load, test_lane_map / test_lane_past ~30 min, test_jtoken_eval ~16 min, test_lane_live ~12 min; CPU only, no GPU job needed).
# usage (from the problem folder): wsl bash tools/maintenance/run_token_tests.sh [LOGDIR]      LOGDIR default temp/token_tests (scratch, ignored by git); exit 1 when any test failed. Last run: 2026-11-17 / 18, 17 of 18 PASS (the failure was a not yet committed file of another chat).
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
printf '%s\n' test_jtoken_eval.py test_jtoken.py test_jmap_train.py test_jpast_train.py test_jlive_train.py test_lane_live.py test_lane_map.py test_jtok.py test_jtok2.py test_jrecur.py test_lane_past.py test_lane_tokcache.py test_jpast.py test_jmap.py test_build_tokcache_root.py test_lane_suite.py test_token_readout.py test_token_time_ens.py test_jtoken_fast.py test_lane_ptok_gpu.py | xargs -P 3 -I{} bash -c 'run1 {}'
cat "$LOG/results.txt"
! grep -q "^FAIL" "$LOG/results.txt"

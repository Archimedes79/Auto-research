#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# Run every self-test of the project (about 10 minutes, uses the GPU for four of them). Usage (from the problem folder): wsl bash tools/maintenance/run_all_tests.sh
R=$BFLC
P="bash $R/tools/env/_py14.sh"
fail=0
run() {
  name="$1"; shift
  out=$($P "$@" 2>&1); rc=$?
  last=$(echo "$out" | tail -1 | cut -c1-110)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; else echo "FAIL  $name  | $last"; fail=1; fi
}
run "grid metric reproduces stored numbers" lane_grid/selftest.py
run "graph metric (identity, DP sparsity)"  sparse_lanes/selftest.py
run "connectivity / duplicate / range metrics" topo_metrics.py
run "metric response to controlled errors"  test_metric_response.py
run "evaluate.py reproduces the golden numbers (cached, all registry names)" evaluate.py --golden check --cached-only
run "builder spec tests (18)"               test_p16_spec.py
run "trainer: crash + resume, NaN, rollback" trainer_selftest.py
run "autoresearch tools (launcher, git tag, ledger, schedule, decide)" $REPO/autoresearch/tools/selftest.py
run "structure and documents (check_structure)" $REPO/autoresearch/tools/check_structure.py --selftest
run "images + video tools"                  viz/selftest.py
run "champion fast path == reference graphs" p16_fast_check.py --n 10
run "registration K188 (synthetic shifts, gradient)" p18_regis.py
run "numba union == FastUnion graphs (120 frames)" p18_asm_check.py --n 120 --reps 1
exit $fail

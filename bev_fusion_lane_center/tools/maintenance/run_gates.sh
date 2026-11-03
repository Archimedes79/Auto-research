#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# The fast gates (the BEV system's tests/run_gates.sh, adapted): the tests that need no GPU and no stored network output, a few minutes. Run them before every commit of code.
# The full suite (metrics, golden numbers, the chain, the tools) is run_all_tests.sh (about 25 minutes, uses the GPU for some).
# Usage (from the problem folder): wsl bash tools/maintenance/run_gates.sh
R=$BFLC
P="bash $R/tools/env/_py14.sh"
fail=0
run() {
  name="$1"; shift
  out=$($P "$@" 2>&1); rc=$?
  last=$(echo "$out" | tail -1 | cut -c1-110)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; else echo "FAIL  $name  | $last"; fail=1; fi
}
run "trainer: crash + resume, NaN, rollback, trainsuite contract (log.jsonl, stop = exit 75)" trainer_selftest.py
run "builder spec tests"                    test_p16_spec.py
run "output stage simplify_dp"              test_dp_simplify.py
run "TEST guard + cache key of evaluate.py" test_evaluate_guard.py
run "evaluate.py parts (stored inputs, CPU)" test_eval_parts.py
run "lane task of the evaluation suite"     test_lane_task.py
run "route data: one copy of resample_matrix / route_item" test_route_data.py
run "seed ring + output union"              test_seed_ring_union.py
run "start check of a training config (startable)" startable.py
run "config -> trainer command line (run_config)" run_config.py --selftest
run "graph metric (identity, DP sparsity)"  sparse_lanes/selftest.py
run "grid metric reproduces stored numbers" lane_grid/selftest.py
exit $fail

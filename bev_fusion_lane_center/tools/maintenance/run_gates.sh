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
run "trainer: crash + resume, NaN, rollback, Training contract (log.jsonl, stop = exit 75)" trainer_selftest.py
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
# the lane head of E273 and later (CPU, synthetic frames; the tests on stored lane GT frames are in run_all_tests.sh)
run "head: targets, nodes, links, Hermite oracle"        grid/qtargets.py
run "head: prediction contract"                          grid/qformat.py
run "head: map simulator"                                grid/map_sim.py
run "head: junction ends and pairs of the lane GT (E284)" grid/qends.py
run "head: end head, loss, counts (E284)"                grid/qendnet.py
run "head: end read-out (connectors, rule) (E284)"       qends_eval.py selftest
run "head: junction GT layer, order rule (E285)"         grid/jgt.py
run "head: junction map targets (E285)"                  grid/qjunction.py
run "head: junction maps head, loss, keypoints (E286)"   grid/jmaps.py
run "head: junction maps scoring (E286)"                 jmaps_eval.py selftest
run "head: junction pictures (E285)"                     viz/junction_view.py --selftest
run "head: pair head v2, order penalty (E287)"            grid/jends.py
run "head: pair head v2 read-out (E287)"                 jends_eval.py selftest
run "head: rule control of the pair decision (E287)"      jrule_control.py selftest
run "head: read-out replay (E288)"                       readout_replay.py selftest
run "head: failure splits of the pair head (E300)"        jsplits.py selftest
run "head: pair precision at a fixed recall (E302)"       jpair_pr.py selftest
run "head: torch Matrix-NMS layer (E292)"                 grid/qmnms.py
run "head: all phases, gradients, E274 switches"         grid/qgraph.py
run "head: pixel targets"                                grid/qraster.py
run "head: tier loss switches (defaults = E273 loss)"    grid/test_tier_loss.py
run "head: close-node read-out (rule cases, 300 frames)" test_qgraph_readout.py
run "head: scorer glue (toy lane)"                       test_qgraph_score_glue.py
run "head: tier metrics"                                 test_tier_metrics.py
run "head: pictures selftest"                            viz/qgraph_view.py --selftest
exit $fail

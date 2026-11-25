#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# The fast gates (the BEV system's tests/run_gates.sh, adapted): the tests that need no GPU, a few minutes. A few read stored run data (runs/, data/): in a fresh checkout they print SKIP with the missing path, not FAIL. Run them before every commit of code.
# The full suite (metrics, golden numbers, the chain, the tools) is run_all_tests.sh (about 25 minutes, uses the GPU for some).
# Usage (from the problem folder): wsl bash tools/maintenance/run_gates.sh
R=$BFLC
P="bash $R/tools/env/_py14.sh"
fail=0; npass=0; nskip=0; nfail=0
# PASS / FAIL as before; SKIP = the gate reads stored data that is not in git (runs/, data/, models/: run folders, caches, checkpoints): a fresh checkout skips it with the missing path named
# (the same rule as run_all_tests.sh); any other failure is a FAIL
run() {
  name="$1"; shift
  out=$($P "$@" 2>&1); rc=$?
  last=$(echo "$out" | tail -1 | cut -c1-110)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; npass=$((npass+1))
  elif echo "$out" | grep -qE "(FileNotFoundError|No such file or directory).*/(runs|data|models)/"; then
    miss=$(echo "$out" | grep -oE "(FileNotFoundError|No such file or directory)[^/]*'?/[^ ']*/(runs|data|models)/[^ ']*" | head -1 | grep -oE "/[^ ']*/(runs|data|models)/[^ ']*" | sed 's#.*/\(runs\|data\|models\)/#\1/#' | head -1)
    echo "SKIP  $name  | needs data that is not in git: ${miss:-see the message}"; nskip=$((nskip+1))
  else echo "FAIL  $name  | $last"; fail=1; nfail=$((nfail+1)); fi
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
run "head: targets, nodes, links, Hermite oracle"        net/prep/qtargets.py
run "head: prediction contract"                          net/model/qformat.py
run "head: map simulator"                                net/prep/map_sim.py
run "head: junction ends and pairs of the lane GT (E284)" net/prep/qends.py
run "head: end head, loss, counts (E284)"                net/model/qendnet.py
run "head: end read-out (connectors, rule) (E284)"       qends_eval.py selftest
run "head: junction GT layer, order rule (E285)"         net/prep/jgt.py
run "head: junction map targets (E285)"                  net/prep/qjunction.py
run "head: junction maps head, loss, keypoints (E286)"   net/model/jmaps.py
run "head: pair head v2, order penalty (E287)"            net/model/jends.py
run "head: pair head v2 read-out (E287)"                 jends_eval.py selftest
run "head: rule control of the pair decision (E287)"      jrule_control.py selftest
run "head: pair precision at a fixed recall (E302)"       jpair_pr.py selftest
run "head: maneuver coverage of a delivered graph (E310)"   jmaneuver.py selftest
run "head: diagnostics of the junction thread (E307 - E313)"  jdiag.py selftest
run "head: wiring of the connectors, oracle gate (E322)"  jwiring.py selftest
run "head: previous-keyframe ceiling and replay (E299 step 0)"  jtime.py selftest
run "head: wiring twin of the nearest-piece rule, kept ends (E325)"  net/model/jwire.py
run "head: wiring head Wire, features, labels, places, leak tests (E325)"  net/model/jwire_head.py
run "head: wiring head data, records, agreement (E325)"  jwire_data.py selftest
run "head: wiring head training, ablation (E325)"  jwire_train.py selftest
run "head: wiring head checks: twin card and live records, planted differences (E350)"  jwire_check.py selftest
run "head: wiring head verdict: sign classes, kill, shift flag, planted cards (E350)"  jwire_verdict.py selftest
run "timing: junction system ms per frame, section markers on a toy JEnds (E342 part B, CPU)"  jtiming.py selftest
run "timing: CUDA graph of the junction front, equality driver, dump directories, score dry check, planted differences (E351, CPU)"  jgraph.py selftest
run "head: torch Matrix-NMS layer (E292)"                 net/model/qmnms.py
run "head: Matrix-NMS switch, phase 9, mutations (E292)"  net/tests/test_mnms_layer.py
run "head: all phases, gradients, E274 switches"         net/model/qgraph.py
run "head: pixel targets"                                net/prep/qraster.py
run "head: tier loss switches (defaults = E273 loss)"    net/tests/test_tier_loss.py
run "head: close-node read-out (rule cases, 300 frames)" test_qgraph_readout.py
run "head: pictures selftest"                            viz/qgraph_view.py --selftest
echo "gates: $npass PASS, $nskip SKIP (stored data not in this checkout), $nfail FAIL"
exit $fail

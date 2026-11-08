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
run "TEST guard + cache key of evaluate.py"  test_evaluate_guard.py
run "evaluate.py parts: route success, failure series, timing (stored inputs, CPU)" test_eval_parts.py
run "lane task of the evaluation suite (moved from autoresearch, D24)" test_lane_task.py
run "p18_joint guarded step: non-finite loss, skipped optimiser steps, scale-drop rule" test_p18_joint_guard.py
run "evaluate.py reproduces the golden numbers (cached, all registry names)" evaluate.py --golden check --cached-only
run "builder spec tests (18)"               test_p16_spec.py
run "output stage simplify_dp (10)"         test_dp_simplify.py
run "route data: one copy of resample_matrix / route_item" test_route_data.py
run "seed ring (E266) + output union (E264)" test_seed_ring_union.py
run "trainer: crash + resume, NaN, rollback" trainer_selftest.py
run "autoresearch tools (launcher, git tag, ledger, schedule, decide)" $REPO/autoresearch/tools/selftest.py
run "structure and documents (check_structure)" $REPO/autoresearch/tools/check_structure.py --selftest
run "agent copies in .claude/agents in step"   $REPO/autoresearch/tools/install_agents.py --check
run "images + video tools"                  viz/selftest.py
run "lane prototype: JSON graph == evaluate's graph, card reproduced" test_lane_prototype.py
run "champion fast path == reference graphs" p16_fast_check.py --n 10
run "registration K188 (synthetic shifts, gradient)" p18_regis.py
run "numba union == FastUnion graphs (120 frames)" p18_asm_check.py --n 120 --reps 1
run "grid chain parts (soft_lines, hold, assign, targets, head, losses; E271 / E272)" grid/soft_lines.py
run "grid hold: ego warp reproduces the second observation" grid/hold.py
run "grid assign: soft assignment with a dustbin" grid/assign.py
run "grid targets (cell targets, heat)" grid/targets.py
run "grid head (2.05 M, reads x[:39])" grid/head.py
run "grid losses" grid/losses.py
run "grid chain: one function, gradient reaches every block (G2)" grid/chain.py
run "grid head spec test (G1) on real frames" grid/test_grid.py
run "grid oracle (G0): the GT through the chain reproduces itself" grid/test_oracle.py
run "E273 lane head: targets, nodes, links, Hermite oracle (synthetic)" grid/qtargets.py
run "E273 lane head: prediction contract" grid/qformat.py
run "E273 lane head: map simulator" grid/map_sim.py
run "E273 lane head: pyramid head" grid/pyramid.py
run "E273 lane head: node head" grid/qnodes.py
run "E273 lane head: link head" grid/qlinks.py
run "E273 lane head: sections" grid/sections.py
run "E273 lane head: map encoder + assignment" grid/qassign.py
run "E273 / E274 lane head: all phases, gradients, E274 switches" grid/qgraph.py
run "E273 pixel targets (synthetic)" grid/qraster.py
run "E273 pixel targets on real lane GT frames" grid/test_qraster.py
run "E273 targets: the oracle on real lane GT frames" grid/test_qtargets.py
run "E273 local scorer (oracle, degradations, lane suite reproduction)" test_qgraph_eval.py
run "E273 pictures: selftest" viz/qgraph_view.py --selftest
run "E274 verdict rule (fake cards)" test_verdict274.py
run "E275 close-node suppression (cases, 300 random frames, toy scorer)" test_qgraph_readout.py
run "E275 verdict rule (fake cards)" test_verdict275.py
run "E275 scorer glue (toy lane, CPU: cards, kill criterion, reproduction gate, dumps)" test_qgraph_score_glue.py
run "E276 tier metrics and verdict276 (toy lane, fake cards)" test_tier_metrics.py
run "E276 tier / vis / sparse targets (toy lane, 40 real frames)" grid/test_tiers_sparse.py
run "E276 tier loss switches (defaults = the frozen E273 loss)" grid/test_tier_loss.py
run "E277 soft close-node step (hard limit on 183 random frames, chain, ties, gradients, model switch, load_model)" grid/test_soft_nms.py
run "E277 verdict rule (fake cards)" test_verdict277.py
run "E282 zones of the sparse node target, zone-matching loss (toy lane, winner rules, E273 loss bit for bit, 6 real frames)" grid/test_zones.py
run "E282 zone metrics (hand cases, position membership, additivity, bootstrap, matched threshold, density against the scorer)" test_zone_metrics.py
run "E282 pilot runner (gate outcomes on fake results, control matched density, targets per arm, run names)" train/test_qnode_pilot.py
run "vertex scoring (selftest)" vertex_eval.py selftest
exit $fail

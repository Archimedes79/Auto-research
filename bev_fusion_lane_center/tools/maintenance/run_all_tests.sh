#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# Run every self-test of the project (about 10 minutes, uses the GPU for four of them). Usage (from the problem folder): wsl bash tools/maintenance/run_all_tests.sh
R=$BFLC
P="bash $R/tools/env/_py14.sh"
fail=0; npass=0; nskip=0; nfail=0
# PASS / FAIL as before; SKIP = the test needs stored data that is not in git (runs/, data/, models/: run folders, caches, checkpoints; data/rebuild.json says how each is made): a fresh checkout skips
# these with the missing path named instead of failing; any other failure is a FAIL
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
run "grid metric reproduces stored numbers" lane_grid/selftest.py
run "graph metric (identity, DP sparsity)"  sparse_lanes/selftest.py
run "connectivity / duplicate / range metrics" topo_metrics.py
run "metric response to controlled errors"  test_metric_response.py
run "TEST guard + cache key of evaluate.py"  test_evaluate_guard.py
run "GPU wait gate before the dump / score steps (D100, CPU)" test_gpu_gate.py
run "pyramid change read-out of E393 (CPU, toy pyramids)" test_pyr_change.py
run "E345 pruning probe: cost table = FlopCounterMode part by part, saved cost of a set (random init)" jprune_cost.py selftest
run "E345 pruning probe: unit table (45 units, 84 points), attention-head edit, hooks, dead-hook assert (random init, CPU)" jprune_probe.py selftest
run "E352 step 0 probe: point table on a random-init model, whole groups, the shim around B's QGraph (random init, CPU, no data)" jprune_b.py selftest
run "E345 pruning probe on the stored E312 / E324 s models: dump == jends_eval.dump, calibration, appliers, controls, saved cost vs narrowed copies, table logic (CPU, about 6 minutes)" test_jprune_probe.py
run "E352 step 0: B-side pruning probe (point table, shim, the empty set = the stored dumps, edited passes, point folder, scoring lines; CPU, about 10 minutes)" test_jprune_b.py
run "E353 narrowing of the junction maps head: groups, equality with the hook-ablated model, MACs = cost table, spec round trip, dump wrapper, trainer (frozen set, run-folder rule, records, resume, n1z, cp), timing machinery, paired timing (CPU, about 3 minutes)" grid/test_jnarrow.py
run "E321 / E343 TEST module (gate, split rows, paired bootstrap, stored-card match, variants)" jtest_eval.py selftest
run "E343 TEST module on toy data (write helper, dump validation, reading rule, scoring glue, control checks, E321 hashes)" test_jtest_eval.py
run "evaluate.py parts: route success, failure series, timing (stored inputs, CPU)" test_eval_parts.py
run "lane task of the evaluation suite (moved from autoresearch, D24)" test_lane_task.py
run "p18_joint guarded step: non-finite loss, skipped optimiser steps, scale-drop rule" test_p18_joint_guard.py
run "evaluate.py reproduces the golden numbers (cached, all registry names)" evaluate.py --golden check --cached-only
run "builder spec tests (18)"               test_p16_spec.py
run "output stage simplify_dp (10)"         test_dp_simplify.py
run "route data: one copy of resample_matrix / route_item" test_route_data.py
run "seed ring (E266) + output union (E264)" test_seed_ring_union.py
run "trainer: crash + resume, NaN, rollback" trainer_selftest.py
# the tools of autoresearch/ run with the system python3 of the WSL (as the hook, the launcher and their documentation call them), not with the conda env: the guard hook's selftest
# expects the RecursionError of Python 3.10 on a 5000-deep event, Python 3.12 of the env parses it (the hook itself is run by python3)
runsys() { local keep="$P"; P="python3"; run "$@"; P="$keep"; }
# runneeds <path under the problem folder> <name> <script> ...: a test of real stored data; SKIP (naming the path) when that data is not in this checkout (a fresh clone), else as run
runneeds() { local need="$1"; shift; if [ -e "$R/$need" ]; then run "$@"; else echo "SKIP  $1  | needs data that is not in git: $need"; nskip=$((nskip+1)); fi; }
runsys "autoresearch tools (launcher, git tag, ledger, schedule, decide)" $REPO/autoresearch/tools/selftest.py
runsys "structure and documents (check_structure)" $REPO/autoresearch/tools/check_structure.py --selftest
runsys "recipe.py: repeat a run from a clone (base + overlay, show / repeat / check / store on throw-away repositories)" $REPO/autoresearch/tools/recipe.py --selftest
if [ -d "$REPO/.claude/agents" ]; then
  runsys "agent copies in .claude/agents in step"   $REPO/autoresearch/tools/install_agents.py --check
else
  echo "SKIP  agent copies in .claude/agents in step  | not installed in this checkout: python3 autoresearch/tools/install_agents.py"; nskip=$((nskip+1))
fi
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
run "E319 junction maps head: heading-guided exchange (flag off = before, zero start, gradients, geometry)" grid/test_jmaps_dirx.py
run "E332 junction maps head without auxiliary maps (flags off = before, bit-identical reduction, loss terms, gradients)" grid/test_jmaps_simplify.py
run "E330 cost of the junction system per block (parameters, MACs)" jmodel_cost.py selftest
run "E331 - E335 adoption rule of a removal arm" jrule_removal.py selftest
runneeds ../data/autogt/lane "E273 pixel targets on real lane GT frames" grid/test_qraster.py
runneeds ../data/autogt/lane "E273 targets: the oracle on real lane GT frames" grid/test_qtargets.py
run "E273 local scorer (oracle, degradations, lane suite reproduction)" test_qgraph_eval.py
run "E273 pictures: selftest" viz/qgraph_view.py --selftest
run "E275 close-node suppression (cases, 300 random frames, toy scorer)" test_qgraph_readout.py
run "E276 tier / vis / sparse targets (toy lane, 40 real frames)" grid/test_tiers_sparse.py
run "E276 tier loss switches (defaults = the frozen E273 loss)" grid/test_tier_loss.py
run "E277 soft close-node step (hard limit on 183 random frames, chain, ties, gradients, model switch, load_model)" grid/test_soft_nms.py
runneeds runs/kept/e276/predfull_e273_p_s0.pkl "E292 Matrix-NMS layer = matrix_nms on 20 frames of B's real dump (kill criterion), switch, phase 9, mutations" grid/test_mnms_layer.py --real
run "E282 zones of the sparse node target, zone-matching loss (toy lane, winner rules, E273 loss bit for bit, 6 real frames)" grid/test_zones.py
run "vertex scoring (selftest)" vertex_eval.py selftest
run "lanegraph champion code == original modules (exact, 24 frames x 2 seeds, read-out fuzz, golden digests; CPU, about 2 minutes)" lanegraph/tests/run_lg_tests.py
# the token head (E365 - E383): none of these was in this list before 2026-11-17
run "E365 token cache: build / read / selftest" grid/test_lane_tokcache.py
run "E365 token head: pair geometry, gate, targets (jtok)" grid/test_jtok.py
run "E365 - E379 token head: likelihood by hand, options, trainer, resume, KD (CPU, synthetic)" grid/test_jtoken.py
run "token head read-out: dumps, sweep, cells, cards, paired scores (toy cache)" test_jtoken_eval.py
run "token head cost / time ensemble" test_token_time_ens.py
run "E372 live token producer (B' in the step) and the live trainer" grid/test_lane_live.py
runneeds runs/e365_cache "E372 live training step" grid/test_jlive_train.py
run "jarm: one arm of a token experiment (card words, cells, table)" jarm.py selftest
run "E389 route-association head + trainer (CPU, synthetic, 67 mutants)" grid/test_jroute.py
run "E389 route-association read-out: hand frames, cost, dump / score / rules on 24 real VAL frames (CPU; the full run with the textual mutants is test_jroute_eval.py without argument, about 30 min)" test_jroute_eval.py selftests
runneeds runs/e365_cache "E389 route data: tier-1 route, band labels, forks, map-error modes (needs the VAL cache; mutants: run the file without the argument)" grid/test_lane_route.py selftests
run "E390 SD road-graph encoding: leaks, invariance, divider line, paired levels, P6, cut (CPU, hand graphs, mutants)" grid/test_sdgraph.py
run "E394 map-error catalogue (nested draws, per-mode invariants, the TRAIN mix, 10 mutants; CPU, hand graph)" grid/test_maperror.py
run "E394 v1 gate g: the label, the four label-free frame features by hand, the logistic and its TRAIN threshold, the card levels (CPU, no data)" test_maptrust_gate.py
run "E391b existence gate and the delivered-mask rerun: features by hand, per-kind rates, re-weighting, threshold, stop rule, outcome table, the choice and the verdict of part B (CPU, toy data, no stored data)" mapassoc_exist.py selftest
run "E391 map association: perception untouched (no-detach mutant), the label rule, targets + every arm on a real batch (CPU)" grid/test_mapassoc.py
run "pmap: the order-keeping fork-pool map of the CPU scorers" evaluation/pmap.py
run "jthresh: learned decode thresholds (E388): soft F1 by hand, a known optimum, mutants" jthresh.py selftest
run "lane suite of the evaluation (5 checks, about 9 minutes)" test_lane_suite.py
run "simplify_stat: the simplification table of a report from git" tools/simplify_stat.py selftest
run "qual_frames: the qualitative frame pictures (colour bands, one picture through render and the command line; CPU, no dataset needed)" tools/viz/qual_frames.py --selftest
runsh() {                                                                  # a shell tool's selftest (bash instead of _py14.sh)
  name="$1"; shift
  out=$(bash "$@" 2>&1); rc=$?
  last=$(echo "$out" | tail -1 | cut -c1-110)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; npass=$((npass+1)); else echo "FAIL  $name  | $last"; fail=1; nfail=$((nfail+1)); fi
}
runsh "head_check: the clean-copy check of a commit (syntax + one real import; the full check: head_check.sh <ref>)" "$R/tools/maintenance/head_check/head_check.sh" --selftest
echo "== $npass passed, $nskip skipped (need data that is not in git: bev_fusion_lane_center/data/README.md and data/rebuild.json say how it is made), $nfail failed"
exit $fail

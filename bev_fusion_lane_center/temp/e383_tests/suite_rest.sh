#!/bin/bash
# the tests of run_all_tests.sh that did not run on 2026-11-18 (the script file was changed by another chat at 00:41 while it ran: bash stopped reading it after the E276 tier loss switches)
R=/mnt/c/Auto-research/bev_fusion_lane_center
P="bash $R/tools/env/_py14.sh"
REPO=/mnt/c/Auto-research
fail=0
run() {
  name="$1"; shift
  out=$($P "$@" 2>&1); rc=$?
  last=$(echo "$out" | tail -1 | cut -c1-110)
  if [ $rc -eq 0 ]; then echo "PASS  $name  | $last"; else echo "FAIL  $name  | $last"; fail=1; fi
}
run "E277 soft close-node step (hard limit on 183 random frames, chain, ties, gradients, model switch, load_model)" grid/test_soft_nms.py
run "E277 verdict rule (fake cards)" test_verdict277.py
run "E292 Matrix-NMS layer = matrix_nms on 20 frames of B's real dump (kill criterion), switch, phase 9, mutations" grid/test_mnms_layer.py --real
run "E282 zones of the sparse node target, zone-matching loss (toy lane, winner rules, E273 loss bit for bit, 6 real frames)" grid/test_zones.py
run "E282 zone metrics (hand cases, position membership, additivity, bootstrap, matched threshold, density against the scorer)" test_zone_metrics.py
run "E282 pilot runner (gate outcomes on fake results, control matched density, targets per arm, run names)" train/test_qnode_pilot.py
run "vertex scoring (selftest)" vertex_eval.py selftest
run "lanegraph champion code == original modules (exact, 24 frames x 2 seeds, read-out fuzz, golden digests; CPU, about 2 minutes)" lanegraph/tests/run_lg_tests.py
# the token head (E365 - E383): none of these was in this list before 2026-11-17
run "E365 token cache: build / read / selftest" grid/test_lane_tokcache.py
run "E365 token head: pair geometry, gate, targets (jtok)" grid/test_jtok.py
run "E365 token head (jtok2 parts)" grid/test_jtok2.py
run "E365 - E379 token head: likelihood by hand, options, trainer, resume, KD (CPU, synthetic)" grid/test_jtoken.py
run "token head read-out: dumps, sweep, cells, cards, paired scores (toy cache)" test_jtoken_eval.py
run "token head read-out helpers (graphs, per-frame table)" test_token_readout.py
run "token head cost / time ensemble" test_token_time_ens.py
run "E372 live token producer (B' in the step) and the live trainer" grid/test_lane_live.py
run "E372 live training step" grid/test_jlive_train.py
run "token speed path: same loss / gradients / weights as the plain head, real batches below and above the key-weight switch" grid/test_jtoken_fast.py
run "token loader on the GPU: same batches and epoch order as the DataLoader" grid/test_lane_ptok_gpu.py
run "token cache into another root" grid/test_build_tokcache_root.py
run "jarm: one arm of a token experiment (card words, cells, table)" jarm.py selftest
run "pmap: the order-keeping fork-pool map of the CPU scorers" evaluation/pmap.py
exit $fail
